"""MLP-Mixer structures of the frozen registry (the preregistration in force via lb.prereg, sections architecture and registry, unchanged since v1).

Structures: shared, two_groups, reversible_shared (tied class); untied, reversible_untied, wide_single (untied class).
build_pointer_shared() is the sanity construct (untied code path, the same block object held T times); not in the registry.
Reversible stacks use a custom autograd.Function that stores only the stack output and reconstructs coupling inputs in
backward. Checkpointing (set_checkpointing) wraps each non-reversible block application; reversible structures ignore it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F_
from torch.utils.checkpoint import checkpoint

TIED_CLASS = ("shared", "two_groups", "reversible_shared")
UNTIED_CLASS = ("untied", "reversible_untied", "wide_single")
STRUCTURES = TIED_CLASS + UNTIED_CLASS
REVERSIBLE = ("reversible_shared", "reversible_untied")


class MixerBlock(nn.Module):
    """Pre-LayerNorm Mixer block: u = z + TokMLP(LN1(z)); v = u + ChMLP(LN2(u))."""

    def __init__(self, n_tokens, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.tok1 = nn.Linear(n_tokens, n_tokens)
        self.tok2 = nn.Linear(n_tokens, n_tokens)
        self.ln2 = nn.LayerNorm(d)
        self.ch1 = nn.Linear(d, 2 * d)
        self.ch2 = nn.Linear(2 * d, d)

    def forward(self, z):
        t = self.ln1(z).transpose(1, 2)                      # [B, d, n]
        u = z + self.tok2(F_.gelu(self.tok1(t))).transpose(1, 2)
        return u + self.ch2(F_.gelu(self.ch1(self.ln2(u))))


class Coupling(nn.Module):
    """Additive coupling on channel halves: y1 = x1 + F(x2); y2 = x2 + G(y1), F(z) = MixerBlock_{d/2}(z) - z."""

    def __init__(self, n_tokens, d):
        super().__init__()
        if d % 2:
            raise ValueError(f"reversible coupling needs even d, got {d}")
        self.F = MixerBlock(n_tokens, d // 2)
        self.G = MixerBlock(n_tokens, d // 2)

    def f(self, z):
        return self.F(z) - z

    def g(self, z):
        return self.G(z) - z

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=-1)
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return torch.cat([x1, x2], dim=-1)


def _unique_params(modules):
    seen, out = set(), []
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


class _ReversibleStack(torch.autograd.Function):
    """Forward without saving activations; backward reconstructs each coupling input from its output and recomputes
    the coupling locally. Parameter gradients of a coupling used several times are summed over its uses."""

    @staticmethod
    def forward(ctx, x, seq, *params):
        with torch.no_grad():
            y = x
            for c in seq:
                y = c(y)
        ctx.seq = seq
        ctx.params = params
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, gy):
        (y,) = ctx.saved_tensors
        index = {id(p): i for i, p in enumerate(ctx.params)}
        pgrads = [None] * len(ctx.params)
        y = y.detach()
        for c in reversed(ctx.seq):
            with torch.no_grad():
                x = c.inverse(y)
            local = _unique_params([c])
            with torch.enable_grad():
                xr = x.detach().requires_grad_(True)
                yr = c(xr)
                grads = torch.autograd.grad(yr, [xr] + local, gy, allow_unused=True)
            gy = grads[0]
            for p, g in zip(local, grads[1:]):
                if g is None:
                    continue
                i = index[id(p)]
                pgrads[i] = g if pgrads[i] is None else pgrads[i] + g
            y = x
        pgrads = [torch.zeros_like(p) if g is None else g for p, g in zip(ctx.params, pgrads)]
        return (gy, None, *pgrads)


class Head(nn.Module):
    def __init__(self, d, classes):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc = nn.Linear(d, classes)

    def forward(self, z):
        return self.fc(self.ln(z).mean(dim=1))


class MixerNet(nn.Module):
    """embed -> blocks applied in `schedule` order -> head. `blocks` holds unique modules except in the pointer-shared
    sanity construct, where it holds the same block object T times."""

    def __init__(self, task_cfg, structure, d, T, make_blocks, tied):
        super().__init__()
        self.structure, self.d, self.T, self.tied = structure, int(d), int(T), tied
        self.reversible = structure in REVERSIBLE
        self.checkpointing = False
        self.custom_backward = True     # reversible only: False = plain autograd reference (tests)
        self.embed = nn.Linear(task_cfg["token_dim"], d)
        blocks, schedule = make_blocks()
        self.blocks = nn.ModuleList(blocks)
        self.schedule = list(schedule)
        self.head = Head(d, task_cfg["classes"])
        self.n_params = sum(p.numel() for p in self.parameters())

    def set_checkpointing(self, on):
        self.checkpointing = bool(on)
        return self

    def stack(self, z):
        if self.reversible:
            seq = [self.blocks[i] for i in self.schedule]
            if self.custom_backward and torch.is_grad_enabled():
                return _ReversibleStack.apply(z, seq, *_unique_params(seq))
            for c in seq:
                z = c(z)
            return z
        for i in self.schedule:
            blk = self.blocks[i]
            if self.checkpointing and torch.is_grad_enabled():
                z = checkpoint(blk, z, use_reentrant=False)
            else:
                z = blk(z)
        return z

    def forward(self, x):
        return self.head(self.stack(self.embed(x)))


def build_model(task_cfg, structure, d, T):
    if structure not in STRUCTURES:
        raise ValueError(f"unknown structure {structure!r}")
    d, T, n = int(d), int(T), int(task_cfg["n_tokens"])
    if T < 1:
        raise ValueError("T must be >= 1")
    if structure == "wide_single" and T != 1:
        raise ValueError("wide_single has T = 1")
    if structure == "two_groups" and T % 2:
        raise ValueError("two_groups needs even T")

    def make_blocks():
        if structure == "shared":
            return [MixerBlock(n, d)], [0] * T
        if structure == "two_groups":
            a = MixerBlock(n, d)
            b = MixerBlock(n, d)
            return [a, b], [0] * (T // 2) + [1] * (T // 2)
        if structure in ("untied", "wide_single"):
            return [MixerBlock(n, d) for _ in range(T)], list(range(T))
        if structure == "reversible_shared":
            return [Coupling(n, d)], [0] * T
        return [Coupling(n, d) for _ in range(T)], list(range(T))   # reversible_untied

    return MixerNet(task_cfg, structure, d, T, make_blocks, tied=structure in TIED_CLASS)


def build_pointer_shared(task_cfg, d, T):
    """Sanity construct: the untied code path whose ModuleList holds ONE MixerBlock object T times.
    Same construction order as build_model(..., "shared", ...) so the same seed gives the same weights."""
    n = int(task_cfg["n_tokens"])

    def make_blocks():
        blk = MixerBlock(n, int(d))
        return [blk] * int(T), list(range(int(T)))

    return MixerNet(task_cfg, "pointer_shared", d, T, make_blocks, tied=True)
