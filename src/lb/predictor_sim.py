"""Symbolic allocation-event simulator for the lifetime-aware accounting (card S3-P; lb.predictor is the API).

Pure arithmetic on shapes: no torch import, no model, autograd, optimizer, profiler, FakeTensor/meta execution or
tracing. The update of lb.train.train_step is replayed op by op as a list of signed byte events (+ allocation,
- free) under the rules below; the prediction is baseline + max prefix, the profiler's reconstruction rule.

Memory model (rules R1-R6; trace evidence in research/s3/predictor_dev/NOTES.md):
  R1 storage: a Buf is freed (one - event) when the last tensor object (T) on it dies; a view holds its base tensor
     alive (autograd ViewInfo keeps the base), detach / _unsafe_view / SavedVariable(output) alias the storage with
     no base. Python-level references (locals, temporaries, tuple elements) are mirrored by explicit hold/drop in the
     same order as CPython drops them; consecutive frees commute for the max prefix, so only the position of each free
     relative to allocations matters.
  R2 forward ops: allocation order and saved tensors of each aten op used by lb.models (linear's three paths, layer
     norm with its contiguous copy, gelu, add/sub, chunk, cat, mean, log_softmax, nll_loss, scalar division).
  R3 engine: nodes run by priority (highest sequence number first, AccumulateGrad first), a node is ready when all its
     incoming edges delivered; per node: apply (allocations), validate_outputs (sum_to of broadcast bias grads), drop
     the incoming grads, release saved tensors, then pass outputs to the next input buffers. Buffer accumulation is in
     place when the old grad is a non-view tensor nobody else holds, else out of place (old + new, both freed).
     AccumulateGrad steals the first grad and adds the later ones in place (the new one is freed).
  R4 checkpoint (use_reentrant=False, preserve_rng_state): 5056-byte RNG state per checkpointed block call; the
     forward keeps no saved tensor (holders only) and the block input via the frame; the first unpack in backward
     recomputes the block (plus a 5056-byte fork_rng copy) up to the last saved tensor (early stop), the unpacked
     tensor dies at the end of its node's apply; the frame (RNG state, block input) dies with its last holder.
  R5 reversible stack: forward under no_grad saving the stack output only; backward replays the custom Function in
     Python order (inverse under no_grad, detached recompute with grad, nested autograd.grad with captured leaf
     grads, out-of-place sum of per-coupling parameter grads, Python local lifetimes across loop iterations).
  R6 update: zero_grad frees; per microbatch gather, forward, loss, backward; clip_grad_norm_ (foreach off);
     single-tensor optimizer step temporaries for adam, sgd_momentum, sgd and Adafactor (Python local lifetimes).
"""
import heapq
import itertools

F32 = 4
I64 = 8
RNG_STATE_BYTES = 5056          # torch.get_rng_state() on CPU: one uint8 tensor of 5056 bytes (mt19937 state)
INF_SEQ = float("inf")


def numel(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def contig_strides(shape):
    st, z = [], 1
    for s in reversed(shape):
        st.append(z)
        z *= s
    return tuple(reversed(st))


def is_contig(shape, strides):
    z = 1
    for s, st in zip(reversed(shape), reversed(strides)):
        if s != 1:
            if st != z:
                return False
            z *= s
    return True


def compute_stride(oldshape, oldstride, newshape):
    """Port of at::detail::computeStride: strides of a view with newshape, or None when a copy is needed."""
    if len(oldshape) == 0:
        return tuple(1 for _ in newshape)
    newstride = [0] * len(newshape)
    view_d = len(newshape) - 1
    chunk_base_stride = oldstride[-1]
    tensor_numel, view_numel = 1, 1
    for tensor_d in range(len(oldshape) - 1, -1, -1):
        tensor_numel *= oldshape[tensor_d]
        if tensor_d == 0 or (oldshape[tensor_d - 1] != 1 and
                             oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride):
            while view_d >= 0 and (view_numel < tensor_numel or newshape[view_d] == 1):
                newstride[view_d] = view_numel * chunk_base_stride
                view_numel *= newshape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if tensor_d > 0:
                chunk_base_stride = oldstride[tensor_d - 1]
                tensor_numel, view_numel = 1, 1
    if view_d != -1:
        return None
    return tuple(newstride)


class StopRecompute(Exception):
    pass


class Buf:
    __slots__ = ("nbytes", "refs")

    def __init__(self, nbytes):
        self.nbytes, self.refs = nbytes, 0


class T:
    __slots__ = ("buf", "shape", "strides", "base", "refs", "grad_fn", "out_nr", "rg", "acc", "grad", "dead", "name")

    @property
    def contig(self):
        return is_contig(self.shape, self.strides)


class Node:
    __slots__ = ("name", "seq", "edges", "in_shapes", "nin", "fn", "saved", "frame", "slots", "var", "info")


class Frame:
    __slots__ = ("rng", "inp", "fn", "nslots", "live", "recomputed", "done", "counter")


class Sim:
    def __init__(self):
        self.ev, self.tags = [], []
        self.phase = ""
        self.grad_mode = True
        self.seq = 0
        self.pack = None        # Frame of the checkpointed forward being recorded
        self.recomp = None      # Frame being recomputed
        self.track = None       # tensors created during a recomputation

    # ---------------------------------------------------------------- storage (R1)
    def emit(self, nb, tag):
        if nb:
            self.ev.append(nb)
            self.tags.append((self.phase, tag))

    def _mk(self, buf, shape, strides, base, track=True):
        t = T()
        t.buf = buf
        buf.refs += 1
        t.shape, t.strides, t.base = tuple(shape), tuple(strides), base
        t.refs, t.grad_fn, t.out_nr, t.rg, t.acc, t.grad, t.dead, t.name = 1, None, 0, False, None, None, False, ""
        if track and self.track is not None:
            self.track.append(t)
        return t

    def new(self, shape, isz=F32, tag="alloc"):
        shape = tuple(shape)
        b = Buf(numel(shape) * isz)
        self.emit(b.nbytes, tag)
        return self._mk(b, shape, contig_strides(shape), None)

    def view(self, t, shape, strides):
        base = t.base if t.base is not None else t
        self.hold(base)
        return self._mk(t.buf, shape, strides, base)

    def alias(self, t, shape=None, track=True):
        """detach / tensor_data / _unsafe_view: shares the storage, keeps no base."""
        shape = t.shape if shape is None else tuple(shape)
        strides = t.strides if shape == t.shape else contig_strides(shape)
        return self._mk(t.buf, shape, strides, None, track)

    @staticmethod
    def hold(t):
        t.refs += 1
        return t

    def drop(self, t):
        if t is None:
            return
        if t.refs <= 0:
            raise AssertionError("simulator: drop of a dead tensor")
        t.refs -= 1
        if t.refs == 0:
            t.dead = True
            t.buf.refs -= 1
            if t.buf.refs == 0:
                self.emit(-t.buf.nbytes, "free")
            if t.base is not None:
                self.drop(t.base)

    def drop_all(self, ts):
        for t in ts:
            self.drop(t)

    def scalar_op_inplace(self):
        """In-place op with a Python scalar: wrapped number (8 bytes) and its float32 copy (4 bytes), both freed."""
        w = self.new((), I64, "wrapped_scalar")
        c = self.new((), F32, "scalar_to")
        self.drop(c)
        self.drop(w)

    # ---------------------------------------------------------------- autograd records (R2, R4)
    def edge(self, x):
        if x is None or not x.rg:
            return None
        if x.grad_fn is not None:
            return (x.grad_fn, x.out_nr)
        if x.acc is None:
            n = Node()
            n.name, n.seq, n.edges, n.in_shapes, n.nin, n.fn = "AccumulateGrad", INF_SEQ, [], [], 1, None
            n.saved, n.frame, n.slots, n.var, n.info = {}, None, [], x, None
            x.acc = n
        return (x.acc, 0)

    def node(self, name, inputs, fn, saved=(), nin=1, info=None):
        if not self.grad_mode or not any(x is not None and x.rg for x in inputs):
            return None
        self.seq += 1
        n = Node()
        n.name, n.seq, n.nin, n.fn, n.info = name, self.seq, nin, fn, info
        n.edges = [self.edge(x) for x in inputs]
        n.in_shapes = [x.shape if x is not None else None for x in inputs]
        n.saved, n.frame, n.slots, n.var = {}, None, [], None
        self.save(n, saved)
        return n

    def save(self, n, items):
        """items: (key, tensor) pairs saved by node n (inputs before the op's compute, outputs after it)."""
        if n is None:
            return
        for key, t in items:
            if t is None:
                continue
            if self.recomp is not None:
                f = self.recomp
                f.recomputed[f.counter] = self.alias(t, track=False)
                f.counter += 1
                if f.counter == f.nslots:
                    raise StopRecompute()
            elif self.pack is not None:
                f = self.pack
                n.frame = f
                n.slots.append((key, f.nslots))
                f.nslots += 1
                f.live += 1
            else:
                n.saved[key] = self.hold(t)

    @staticmethod
    def out(t, n, nr=0):
        if n is not None:
            t.grad_fn, t.out_nr, t.rg = n, nr, True
        return t

    def unpack(self, n):
        """Saved tensors of node n at the start of its apply: {key: T}; checkpointed nodes get owned tensors."""
        if n.frame is None:
            return dict(n.saved), False
        f = n.frame
        if not f.done:
            self.recompute(f)
        return {key: f.recomputed.pop(i) for key, i in n.slots}, True

    def unpack_end(self, sv, owned):
        if owned:
            self.drop_all(sv.values())

    def release(self, n):
        self.drop_all(n.saved.values())
        n.saved = {}
        if n.frame is not None and n.slots:
            f = n.frame
            f.live -= len(n.slots)
            n.slots = []
            if f.live == 0:
                self.drop(f.rng)
                self.drop(f.inp)

    # ---------------------------------------------------------------- checkpoint (R4)
    def checkpoint(self, fn, z):
        f = Frame()
        f.rng = self.new((RNG_STATE_BYTES,), 1, "ckpt_rng_state")
        f.inp = self.hold(z)
        f.fn, f.nslots, f.live, f.recomputed, f.done, f.counter = fn, 0, 0, {}, False, 0
        self.seq += 1                                   # _NoopSaveInputs node
        prev = self.pack
        self.pack = f
        try:
            out = fn(self, z)
        finally:
            self.pack = prev
        return out

    def recompute(self, f):
        rng = self.new((RNG_STATE_BYTES,), 1, "fork_rng_state")
        prev = (self.recomp, self.track, self.grad_mode, self.pack)
        self.recomp, self.track, self.grad_mode, self.pack = f, [], True, None
        f.counter = 0
        track = self.track
        try:
            zin = self.alias(f.inp)
            zin.rg = True
            f.fn(self, zin)
            raise AssertionError("simulator: recomputation did not reach the last saved tensor")
        except StopRecompute:
            pass
        finally:
            self.recomp, self.track, self.grad_mode, self.pack = prev
        self.drop(rng)
        ids = {id(t) for t in track}
        for t in track:                                  # the recomputation's frames and graph die
            if not t.dead:
                t.dead = True
                t.refs = 0
                t.buf.refs -= 1
                if t.buf.refs == 0:
                    self.emit(-t.buf.nbytes, "free")
                if t.base is not None and id(t.base) not in ids:
                    self.drop(t.base)
        f.done = True

    # ---------------------------------------------------------------- engine (R3)
    def sum_to(self, o, shape):
        lead = len(o.shape) - len(shape)
        keep = tuple(1 if i < lead else (1 if shape[i - lead] == 1 else o.shape[i]) for i in range(len(o.shape)))
        k = self.new(keep, tag="sum_to")
        v = self.view(k, shape, contig_strides(shape))
        self.drop(k)
        return v

    def accumulate(self, old, new):
        if old.base is None and old.refs == 1 and old.buf.refs == 1:
            self.drop(new)
            return old
        s = self.new(old.shape, tag="accumulate_out_of_place")
        self.drop(old)
        self.drop(new)
        return s

    def accumulate_grad(self, var, g):
        if var.grad is None:
            if g.refs <= 1:
                var.grad = self.alias(g, track=False)
            else:
                var.grad = self.new(var.shape, tag="grad_clone")
            self.drop(g)
        else:
            self.drop(g)

    def run_engine(self, root_t, seed, capture=None):
        root = root_t.grad_fn
        deps, nodes, stack = {}, {id(root): root}, [root]
        while stack:
            n = stack.pop()
            for e in n.edges:
                if e is None:
                    continue
                m = e[0]
                deps[id(m)] = deps.get(id(m), 0) + 1
                if id(m) not in nodes:
                    nodes[id(m)] = m
                    stack.append(m)
        buffers, heap, cnt = {}, [], itertools.count()
        b = [None] * root.nin
        b[root_t.out_nr] = self.hold(seed)
        buffers[id(root)] = b
        heapq.heappush(heap, (-root.seq, next(cnt), root))
        captured = {}
        while heap:
            _, _, n = heapq.heappop(heap)
            inputs = buffers.pop(id(n))
            if n.name == "AccumulateGrad":
                g = inputs[0]
                if capture is not None:
                    captured[id(n.var)] = g
                elif g is not None:
                    self.accumulate_grad(n.var, g)
                continue
            outs = n.fn(self, n, inputs)
            for i, o in enumerate(outs):                      # validate_outputs
                if o is None:
                    continue
                if n.edges[i] is None:
                    raise AssertionError(f"simulator: {n.name} computed a grad for an input without edge")
                if o.shape != n.in_shapes[i]:
                    outs[i] = self.sum_to(o, n.in_shapes[i])
                    self.drop(o)
            self.drop_all([x for x in inputs if x is not None])
            self.release(n)
            for i, e in enumerate(n.edges):
                if e is None:
                    continue
                m, slot = e
                o = outs[i]
                deps[id(m)] -= 1
                bb = buffers.get(id(m))
                if bb is None:
                    bb = buffers[id(m)] = [None] * m.nin
                if o is not None:
                    bb[slot] = o if bb[slot] is None else self.accumulate(bb[slot], o)
                if deps[id(m)] == 0:
                    heapq.heappush(heap, (-m.seq, next(cnt), m))
        if buffers:
            raise AssertionError("simulator: engine finished with pending buffers")
        return captured

    # ---------------------------------------------------------------- forward ops (R2)
    def t2(self, W):
        """W.t() of a 2-D parameter: a view; TBackward0 passes grad.t() (a view) to W's AccumulateGrad."""
        v = self.view(W, (W.shape[1], W.shape[0]), (W.strides[1], W.strides[0]))
        n = self.node("TBackward0", [W], _bw_transpose2)
        return self.out(v, n)

    def transpose12(self, x):
        s, st = x.shape, x.strides
        v = self.view(x, (s[0], s[2], s[1]), (st[0], st[2], st[1]))
        n = self.node("TransposeBackward0", [x], _bw_transpose12)
        return self.out(v, n)

    def reshape(self, x, shape):
        """x.reshape(shape) in forward: a view (ViewBackward0) when possible, else a contiguous copy
        (CloneBackward0) and _unsafe_view (UnsafeViewBackward0)."""
        st = compute_stride(x.shape, x.strides, shape)
        if st is not None:
            v = self.view(x, shape, st)
            return self.out(v, self.node("ViewBackward0", [x], _bw_reshape))
        c = self.new(x.shape, tag="reshape_copy")
        self.out(c, self.node("CloneBackward0", [x], _bw_pass))
        u = self.alias(c, shape)
        self.out(u, self.node("UnsafeViewBackward0", [c], _bw_reshape))
        self.drop(c)
        return u

    def view_op(self, x, shape):
        st = compute_stride(x.shape, x.strides, shape)
        if st is None:
            raise AssertionError("simulator: view of a non-viewable tensor")
        v = self.view(x, shape, st)
        return self.out(v, self.node("ViewBackward0", [x], _bw_reshape))

    def unsafe_view(self, x, shape):
        u = self.alias(x, shape)
        return self.out(u, self.node("UnsafeViewBackward0", [x], _bw_reshape))

    def addmm(self, bias, mat1, W):
        Wt = self.t2(W)
        saved = []
        if Wt.rg:
            saved.append(("mat1", mat1))
        if mat1.rg:
            saved.append(("mat2", Wt))
        n = self.node("AddmmBackward0", [bias, mat1, Wt], _bw_addmm, saved,
                      info={"mat1_shape": mat1.shape, "mat1_strides": mat1.strides, "mat2_shape": Wt.shape})
        o = self.out(self.new((mat1.shape[0], W.shape[0]), tag="addmm"), n)
        self.drop(Wt)
        return o

    def mm(self, mat1, mat2):
        saved = []
        if mat2.rg:
            saved.append(("self", mat1))
        if mat1.rg:
            saved.append(("mat2", mat2))
        n = self.node("MmBackward0", [mat1, mat2], _bw_mm, saved,
                      info={"mat1_shape": mat1.shape, "mat1_strides": mat1.strides, "mat2_shape": mat2.shape})
        return self.out(self.new((mat1.shape[0], mat2.shape[1]), tag="mm"), n)

    def linear(self, x, W, b):
        if len(x.shape) == 2:
            return self.addmm(b, x, W)
        if x.contig:                                            # _flatten_nd_linear
            xr = self.reshape(x, (numel(x.shape[:-1]), x.shape[-1]))
            r = self.addmm(b, xr, W)
            o = self.view_op(r, tuple(x.shape[:-1]) + (W.shape[0],))
            self.drop(r)
            self.drop(xr)
            return o
        Wt = self.t2(W)                                         # matmul(x, W.t()) folded to mm, then add_(bias)
        t1f = self.reshape(x, (numel(x.shape[:-1]), x.shape[-1]))
        m = self.mm(t1f, Wt)
        o = self.unsafe_view(m, tuple(x.shape[:-1]) + (W.shape[0],))
        self.drop(m)
        self.drop(t1f)
        self.drop(Wt)
        n = self.node("AddBackward0", [o, b], _bw_add)          # in-place add_(bias)
        if n is not None:
            o.grad_fn, o.out_nr = n, 0
        return o

    def layer_norm(self, x, w, b):
        n = self.node("NativeLayerNormBackward0", [x, w, b], _bw_layer_norm, [("input", x), ("weight", w), ("bias", b)],
                      nin=3)
        xc = None if x.contig else self.new(x.shape, tag="ln_contiguous_input")
        o = self.out(self.new(x.shape, tag="ln_out"), n, 0)
        M = numel(x.shape[:-1])
        mean = self.out(self.new((M,), tag="ln_mean"), n, 1)
        rstd = self.out(self.new((M,), tag="ln_rstd"), n, 2)
        if xc is not None:
            self.drop(xc)
        self.save(n, [("mean", mean), ("rstd", rstd)])
        self.drop(mean)
        self.drop(rstd)
        return o

    def gelu(self, x):
        n = self.node("GeluBackward0", [x], _bw_gelu, [("self", x)])
        return self.out(self.new(x.shape, tag="gelu"), n)

    def add(self, a, c):
        n = self.node("AddBackward0", [a, c], _bw_add)
        return self.out(self.new(a.shape, tag="add"), n)

    def sub(self, a, c):
        n = self.node("SubBackward0", [a, c], _bw_sub)
        return self.out(self.new(a.shape, tag="sub"), n)

    def chunk2(self, x):
        s, st = x.shape, x.strides
        h = s[-1] // 2
        n = self.node("SplitBackward0", [x], _bw_split, nin=2, info={"shape": s})
        a = self.out(self.view(x, s[:-1] + (h,), st), n, 0)
        c = self.out(self.view(x, s[:-1] + (h,), st), n, 1)
        return a, c

    def cat2(self, a, c):
        n = self.node("CatBackward0", [a, c], _bw_cat, info={"h": a.shape[-1]})
        return self.out(self.new(a.shape[:-1] + (a.shape[-1] + c.shape[-1],), tag="cat"), n)

    def mean_dim1(self, x):
        n = self.node("MeanBackward1", [x], _bw_mean, info={"shape": x.shape})
        o = self.out(self.new((x.shape[0], x.shape[2]), tag="mean"), n)
        self.scalar_op_inplace()                                # sum then div_ by the count
        return o

    def cross_entropy_sum_div(self, logits, target, b):
        """F_.cross_entropy(logits, target, reduction="sum") / b; drops the logits temporary like the caller."""
        n1 = self.node("LogSoftmaxBackward0", [logits], _bw_log_softmax)
        lsm = self.out(self.new(logits.shape, tag="log_softmax"), n1)
        self.save(n1, [("result", lsm)])
        n2 = self.node("NllLossBackward0", [lsm, target], _bw_nll, [("self", lsm), ("target", target)],
                       info={"shape": lsm.shape})
        nll = self.out(self.new((), tag="nll_out"), n2)
        tw = self.new((), tag="nll_total_weight")
        self.save(n2, [("total_weight", tw)])
        self.drop(tw)
        self.drop(lsm)
        self.drop(logits)
        sc = self.new((), I64, "wrapped_scalar")
        n3 = self.node("DivBackward0", [nll, None], _bw_div, [("other", sc)])
        cv = self.new((), F32, "scalar_to")
        loss = self.out(self.new((), tag="div"), n3)
        self.drop(cv)
        self.drop(sc)
        self.drop(nll)
        return loss


# -------------------------------------------------------------------- backward formulas (R3)
def _bw_pass(sim, n, gs):
    return [sim.hold(gs[0])]


def _bw_transpose2(sim, n, gs):
    g = gs[0]
    return [sim.view(g, (g.shape[1], g.shape[0]), (g.strides[1], g.strides[0]))]


def _bw_transpose12(sim, n, gs):
    g = gs[0]
    s, st = g.shape, g.strides
    return [sim.view(g, (s[0], s[2], s[1]), (st[0], st[2], st[1]))]


def _reshape_bw(sim, g, shape):
    st = compute_stride(g.shape, g.strides, shape)
    if st is not None:
        return sim.view(g, shape, st)
    return sim.new(shape, tag="reshape_copy_bw")


def _bw_reshape(sim, n, gs):
    return [_reshape_bw(sim, gs[0], n.in_shapes[0])]


def _mm_mat1_bw(sim, shape, strides):
    """mm_mat1_backward: column-major mat1 gets a column-major grad (transposed view of a fresh tensor)."""
    if strides[0] == 1 and strides[1] == shape[0] and shape[1] != 1:
        r = sim.new((shape[1], shape[0]), tag="mm_mat1_grad")
        v = sim.view(r, shape, (1, shape[0]))
        sim.drop(r)
        return v
    return sim.new(shape, tag="mm_mat1_grad")


def _mm_mat2_bw(sim, shape):
    """mm_mat2_backward for mat2 = W.t() (column-major): grad.t().mm(mat1).t(), a view of a fresh [out, in] tensor."""
    r = sim.new((shape[1], shape[0]), tag="mm_mat2_grad")
    v = sim.view(r, shape, (1, shape[0]))
    sim.drop(r)
    return v


def _bw_addmm(sim, n, gs):
    g = gs[0]
    sv, owned = sim.unpack(n)
    outs = [None, None, None]
    if n.edges[1] is not None:
        outs[1] = _mm_mat1_bw(sim, n.info["mat1_shape"], n.info["mat1_strides"])
    if n.edges[2] is not None:
        outs[2] = _mm_mat2_bw(sim, n.info["mat2_shape"])
    if n.edges[0] is not None:
        outs[0] = sim.hold(g)
    sim.unpack_end(sv, owned)
    return outs


def _bw_mm(sim, n, gs):
    sv, owned = sim.unpack(n)
    outs = [None, None]
    if n.edges[1] is not None:
        outs[1] = _mm_mat2_bw(sim, n.info["mat2_shape"])
    if n.edges[0] is not None:
        outs[0] = _mm_mat1_bw(sim, n.info["mat1_shape"], n.info["mat1_strides"])
    sim.unpack_end(sv, owned)
    return outs


def _bw_add(sim, n, gs):
    g = gs[0]
    return [sim.hold(g) if e is not None else None for e in n.edges]


def _bw_sub(sim, n, gs):
    g = gs[0]
    outs = [None, None]
    if n.edges[0] is not None:
        outs[0] = sim.hold(g)
    if n.edges[1] is not None:
        outs[1] = sim.new(g.shape, tag="neg")
    return outs


def _bw_gelu(sim, n, gs):
    sv, owned = sim.unpack(n)
    o = sim.new(n.in_shapes[0], tag="gelu_backward")
    sim.unpack_end(sv, owned)
    return [o]


def _bw_layer_norm(sim, n, gs):
    g = gs[0]
    sv, owned = sim.unpack(n)
    x = sv["input"]
    d = x.shape[-1]
    xc = None if x.contig else sim.new(x.shape, tag="lnb_contiguous_input")
    dX = sim.new(x.shape, tag="ln_dX") if n.edges[0] is not None else None
    dw = sim.new((d,), tag="ln_dgamma") if n.edges[1] is not None else None
    db = sim.new((d,), tag="ln_dbeta") if n.edges[2] is not None else None
    gc = None if g.contig else sim.new(g.shape, tag="lnb_contiguous_dY")
    ws = sim.new((2 * d,), tag="lnb_buffer")                   # dgamma/dbeta buffer, one thread
    tiny = sim.new((1,), tag="lnb_resize_tmp")
    sim.drop(tiny)
    sim.drop(ws)
    sim.drop(gc)
    sim.drop(xc)
    sim.unpack_end(sv, owned)
    return [dX, dw, db]


def _bw_split(sim, n, gs):
    o = sim.new(n.info["shape"], tag="split_backward_cat")
    return [o]


def _bw_cat(sim, n, gs):
    g = gs[0]
    h = n.info["h"]
    s, st = g.shape, g.strides
    outs = [None, None]
    if n.edges[0] is not None:
        outs[0] = sim.view(g, s[:-1] + (h,), st)
    if n.edges[1] is not None:
        outs[1] = sim.view(g, s[:-1] + (s[-1] - h,), st)
    return outs


def _bw_mean(sim, n, gs):
    w = sim.new((), I64, "wrapped_scalar")
    c = sim.new((), F32, "scalar_to")
    o = sim.new(n.info["shape"], tag="mean_backward")
    sim.drop(c)
    sim.drop(w)
    return [o]


def _bw_div(sim, n, gs):
    sv, owned = sim.unpack(n)
    c = sim.new((), F32, "scalar_to")
    o = sim.new((), tag="div_backward")
    sim.drop(c)
    sim.unpack_end(sv, owned)
    return [o, None]


def _bw_nll(sim, n, gs):
    sv, owned = sim.unpack(n)
    o = sim.new(n.info["shape"], tag="nll_backward")
    sim.unpack_end(sv, owned)
    return [o, None]


def _bw_log_softmax(sim, n, gs):
    sv, owned = sim.unpack(n)
    o = sim.new(n.in_shapes[0], tag="log_softmax_backward")
    sim.unpack_end(sv, owned)
    return [o]


# -------------------------------------------------------------------- modules (lb.models, same Python order)
def mixer_block(sim, P, z):
    """MixerBlock.forward: t = ln1(z).T; u = z + tok2(gelu(tok1(t))).T; return u + ch2(gelu(ch1(ln2(u))))."""
    a = sim.layer_norm(z, P["ln1.weight"], P["ln1.bias"])
    t = sim.transpose12(a)
    sim.drop(a)
    o1 = sim.linear(t, P["tok1.weight"], P["tok1.bias"])
    g1 = sim.gelu(o1)
    sim.drop(o1)
    o2 = sim.linear(g1, P["tok2.weight"], P["tok2.bias"])
    sim.drop(g1)
    o2t = sim.transpose12(o2)
    sim.drop(o2)
    u = sim.add(z, o2t)
    sim.drop(o2t)
    l2 = sim.layer_norm(u, P["ln2.weight"], P["ln2.bias"])
    h = sim.linear(l2, P["ch1.weight"], P["ch1.bias"])
    sim.drop(l2)
    g2 = sim.gelu(h)
    sim.drop(h)
    c = sim.linear(g2, P["ch2.weight"], P["ch2.bias"])
    sim.drop(g2)
    v = sim.add(u, c)
    sim.drop(c)
    sim.drop(t)
    sim.drop(u)
    return v


def _sub_block(sim, P, z):
    """Coupling.f / Coupling.g: block(z) - z."""
    bz = mixer_block(sim, P, z)
    r = sim.sub(bz, z)
    sim.drop(bz)
    return r


def coupling_forward(sim, C, x):
    x1, x2 = sim.chunk2(x)
    f = _sub_block(sim, C["F"], x2)
    y1 = sim.add(x1, f)
    sim.drop(f)
    g = _sub_block(sim, C["G"], y1)
    y2 = sim.add(x2, g)
    sim.drop(g)
    o = sim.cat2(y1, y2)
    sim.drop_all([x1, x2, y1, y2])
    return o


def coupling_inverse(sim, C, y):
    y1, y2 = sim.chunk2(y)
    g = _sub_block(sim, C["G"], y1)
    x2 = sim.sub(y2, g)
    sim.drop(g)
    f = _sub_block(sim, C["F"], x2)
    x1 = sim.sub(y1, f)
    sim.drop(f)
    o = sim.cat2(x1, x2)
    sim.drop_all([y1, y2, x2, x1])
    return o


def head_forward(sim, P, z):
    l_ = sim.layer_norm(z, P["ln.weight"], P["ln.bias"])
    m = sim.mean_dim1(l_)
    sim.drop(l_)
    o = sim.linear(m, P["fc.weight"], P["fc.bias"])
    sim.drop(m)
    return o


def reversible_stack(sim, net, z):
    """_ReversibleStack.apply(z, seq, *params): no_grad forward saving the output; custom backward (R5)."""
    seq = [net["blocks"][i] for i in net["schedule"]]
    params = []
    for c in seq:
        for p in net["coupling_params"][id(c)]:
            if all(p is not q for q in params):
                params.append(p)
    n = sim.node("ReversibleStackBackward", [z] + params, _bw_reversible, info={"seq": seq, "params": params,
                                                                                 "net": net})
    prev = sim.grad_mode
    sim.grad_mode = False
    y = sim.hold(z)
    for c in seq:
        yn = coupling_forward(sim, c, y)
        sim.drop(y)
        y = yn
    sim.grad_mode = prev
    sim.save(n, [("y", y)])
    return sim.out(y, n)


def _bw_reversible(sim, n, gs):
    seq, params, net = n.info["seq"], n.info["params"], n.info["net"]
    gy = sim.hold(gs[0])
    sv, owned = sim.unpack(n)
    y0 = sim.alias(sv["y"])                       # ctx.saved_tensors
    y = sim.alias(y0)                             # y.detach()
    sim.drop(y0)
    pgrads = [None] * len(params)
    x = xr = yr = g = None
    grads = []
    for c in reversed(seq):
        prev = sim.grad_mode
        sim.grad_mode = False
        xn = coupling_inverse(sim, c, y)
        sim.grad_mode = prev
        sim.drop(x)
        x = xn
        local = net["coupling_params"][id(c)]
        sim.grad_mode = True
        xrn = sim.alias(x)
        xrn.rg = True
        sim.drop(xr)
        xr = xrn
        yrn = coupling_forward(sim, c, xr)
        sim.drop(yr)
        yr = yrn
        cap = sim.run_engine(yr, gy, capture=True)
        sim.grad_mode = prev
        new_grads = [cap.get(id(xr))] + [cap.get(id(p)) for p in local]
        sim.drop_all([t for t in grads if t is not None])
        grads = new_grads
        old_gy = gy
        gy = sim.hold(grads[0])
        sim.drop(old_gy)
        sl = [sim.hold(t) for t in grads[1:]]         # grads[1:] (tuple slice held by the iterator)
        for p, gg in zip(local, sl):
            sim.hold(gg)
            sim.drop(g)
            g = gg
            i = next(k for k, q in enumerate(params) if q is p)
            if pgrads[i] is None:
                pgrads[i] = sim.hold(gg)
            else:
                s = sim.new(p.shape, tag="pgrad_sum")
                sim.drop(pgrads[i])
                pgrads[i] = s
        sim.drop_all(sl)
        yo = y
        y = sim.hold(x)
        sim.drop(yo)
    outs = [sim.hold(gy)] + [sim.hold(t) for t in pgrads]      # return (gy, None, *pgrads)
    sim.drop_all([gy] + pgrads + [y, x, xr, yr] + [t for t in grads if t is not None] + [g])
    sim.unpack_end(sv, owned)
    return outs


def net_forward(sim, net, xc):
    e = sim.linear(xc, net["embed.weight"], net["embed.bias"])
    if net["reversible"]:
        s = reversible_stack(sim, net, e)
    else:
        z = sim.hold(e)
        for i in net["schedule"]:
            P = net["blocks"][i]
            if net["checkpointing"]:
                zn = sim.checkpoint(lambda s_, zz, P=P: mixer_block(s_, P, zz), z)
            else:
                zn = mixer_block(sim, P, z)
            sim.drop(z)
            z = zn
        s = z
    sim.drop(e)
    o = head_forward(sim, net["head"], s)
    sim.drop(s)
    return o


# -------------------------------------------------------------------- optimizer step (R6)
def clip_events(sim, n_grads):
    norms = [sim.new((), tag="clip_norm") for _ in range(n_grads)]
    st = sim.new((n_grads,), tag="clip_stack")
    total = sim.new((), tag="clip_total")
    sim.drop(st)
    sim.drop_all(norms)
    w = sim.new((), I64, "wrapped_scalar")                       # total_norm + 1e-6
    c = sim.new((), F32, "scalar_to")
    s = sim.new((), tag="clip_add")
    sim.drop(c)
    sim.drop(w)
    r = sim.new((), tag="clip_reciprocal")                       # max_norm / (...)
    w = sim.new((), I64, "wrapped_scalar")
    c = sim.new((), F32, "scalar_to")
    coef = sim.new((), tag="clip_coef")
    sim.drop(c)
    sim.drop(w)
    sim.drop(r)
    sim.drop(s)
    cl = sim.new((), tag="clip_clamped")
    sim.drop_all([coef, cl, total])


def adam_step(sim, shapes):
    prev = None
    for shp in shapes:
        sim.scalar_op_inplace()                                  # step_t += 1
        sim.scalar_op_inplace()                                  # exp_avg_sq.mul_(beta2)
        sq = sim.new(shp, tag="adam_sqrt")
        w = sim.new((), I64, "wrapped_scalar")
        c = sim.new((), F32, "scalar_to")
        den = sim.new(shp, tag="adam_denom")
        sim.drop(c)
        sim.drop(w)
        sim.drop(sq)
        sim.scalar_op_inplace()                                  # denom.add_(eps)
        sim.drop(prev)
        prev = den
    sim.drop(prev)


def sgd_momentum_step(sim, shapes):
    for _ in shapes:
        sim.scalar_op_inplace()                                  # buf.mul_(momentum)


def adafactor_step(sim, shapes):
    row_mean = col_mean = var_est = update = grad_sq = None
    for shp in shapes:
        sim.scalar_op_inplace()                                  # step_t += 1
        sim.drop(sim.new((), tag="param_norm"))                  # param.norm(2).item()
        if len(shp) > 1:
            rm = sim.new(shp[:-1] + (1,), tag="row_mean")
            sim.scalar_op_inplace()                              # div_(grad.size(-1))
            sim.drop(row_mean)
            row_mean = rm
            cm = sim.new(shp[:-2] + (1, shp[-1]), tag="col_mean")
            sim.scalar_op_inplace()
            sim.drop(col_mean)
            col_mean = cm
            ve = sim.new(shp, tag="var_estimate_mm")
            sim.drop(var_est)
            var_est = ve
            mn = sim.new(shp[:-2] + (1, 1), tag="row_var_mean")
            sim.scalar_op_inplace()
            sim.drop(mn)
        else:
            gq = sim.new(shp, tag="grad_squared")
            sim.drop(grad_sq)
            grad_sq = gq
            ve = sim.new(shp, tag="variance_clone")
            sim.drop(var_est)
            var_est = ve
        sim.hold(var_est)
        sim.drop(update)
        update = var_est
        sim.drop(sim.new((), tag="update_norm"))                 # update.norm(2).item()
    sim.drop_all([t for t in (row_mean, col_mean, var_est, update, grad_sq) if t is not None])


# -------------------------------------------------------------------- model and update (R6)
BLOCK_PARAMS = ("ln1.weight", "ln1.bias", "tok1.weight", "tok1.bias", "tok2.weight", "tok2.bias",
                "ln2.weight", "ln2.bias", "ch1.weight", "ch1.bias", "ch2.weight", "ch2.bias")
REVERSIBLE = ("reversible_shared", "reversible_untied")


def block_param_shapes(n_tokens, d):
    return {"ln1.weight": (d,), "ln1.bias": (d,), "tok1.weight": (n_tokens, n_tokens), "tok1.bias": (n_tokens,),
            "tok2.weight": (n_tokens, n_tokens), "tok2.bias": (n_tokens,), "ln2.weight": (d,), "ln2.bias": (d,),
            "ch1.weight": (2 * d, d), "ch1.bias": (2 * d,), "ch2.weight": (d, 2 * d), "ch2.bias": (d,)}


def structure_layout(structure, T):
    """(number of unique blocks, schedule) as in lb.models.build_model."""
    if structure == "shared" or structure == "reversible_shared":
        return 1, [0] * T
    if structure == "two_groups":
        if T % 2:
            raise ValueError("two_groups needs even T")
        return 2, [0] * (T // 2) + [1] * (T // 2)
    if structure in ("untied", "wide_single", "reversible_untied"):
        if structure == "wide_single" and T != 1:
            raise ValueError("wide_single has T = 1")
        return T, list(range(T))
    raise ValueError(f"unknown structure {structure!r}")


def param_shapes(n_tokens, token_dim, classes, structure, d, T):
    """Ordered (name, shape) of the unique parameters, in model.parameters() order."""
    nb, _ = structure_layout(structure, T)
    out = [("embed.weight", (d, token_dim)), ("embed.bias", (d,))]
    for b in range(nb):
        if structure in REVERSIBLE:
            if d % 2:
                raise ValueError("reversible coupling needs even d")
            for half in ("F", "G"):
                for k, s in block_param_shapes(n_tokens, d // 2).items():
                    out.append((f"blocks.{b}.{half}.{k}", s))
        else:
            for k, s in block_param_shapes(n_tokens, d).items():
                out.append((f"blocks.{b}.{k}", s))
    out += [("head.ln.weight", (d,)), ("head.ln.bias", (d,)), ("head.fc.weight", (classes, d)),
            ("head.fc.bias", (classes,))]
    return out


def optimizer_state_bytes(optimizer, shapes):
    """Exact bytes of the optimizer state after a step (single-tensor implementations, float32, step tensors)."""
    tot = 0
    for s in shapes:
        n = numel(s) * F32
        if optimizer == "adam":
            tot += 2 * n + F32                                    # exp_avg, exp_avg_sq, step
        elif optimizer == "sgd_momentum":
            tot += n                                              # momentum_buffer
        elif optimizer == "sgd":
            tot += 0
        elif optimizer == "adafactor":
            tot += F32                                            # step
            if len(s) > 1:
                tot += numel(s[:-1]) * F32 + numel(s[:-2] + s[-1:]) * F32    # row_var, col_var
            else:
                tot += n                                          # variance
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}")
    return tot


def _persistent(sim, shape, name=""):
    b = Buf(numel(shape) * F32)
    t = sim._mk(b, shape, contig_strides(shape), None, track=False)
    t.name = name
    return t


def build_net(sim, dims, structure, d, T, checkpointing):
    n_tokens, token_dim, classes = dims
    shapes = param_shapes(n_tokens, token_dim, classes, structure, d, T)
    P = {}
    for name, s in shapes:
        t = _persistent(sim, s, name)
        t.rg = True
        P[name] = t
    nb, schedule = structure_layout(structure, T)
    net = {"embed.weight": P["embed.weight"], "embed.bias": P["embed.bias"], "schedule": schedule,
           "reversible": structure in REVERSIBLE, "checkpointing": bool(checkpointing), "blocks": [],
           "coupling_params": {}, "params": [P[name] for name, _ in shapes],
           "head": {k: P["head." + k] for k in ("ln.weight", "ln.bias", "fc.weight", "fc.bias")}}
    for b in range(nb):
        if structure in REVERSIBLE:
            c = {h: {k: P[f"blocks.{b}.{h}.{k}"] for k in BLOCK_PARAMS} for h in ("F", "G")}
            net["coupling_params"][id(c)] = [c[h][k] for h in ("F", "G") for k in BLOCK_PARAMS]
            net["blocks"].append(c)
        else:
            net["blocks"].append({k: P[f"blocks.{b}.{k}"] for k in BLOCK_PARAMS})
    return net, shapes


def simulate_update(dims, structure, d, T, optimizer, microbatch, checkpointing, batch=64):
    """Event list of the profiled steady-state update. Returns (sim, baseline_bytes, components)."""
    if checkpointing and structure in REVERSIBLE:
        raise ValueError("checkpointing is not combined with reversible structures")
    if batch % microbatch:
        raise ValueError("microbatch must divide the effective batch")
    n_tokens, token_dim, classes = dims
    sim = Sim()
    net, shapes = build_net(sim, dims, structure, d, T, checkpointing)
    params = net["params"]
    for p in params:                                              # gradients from update 1
        p.grad = _persistent(sim, p.shape)
    W = sum(numel(s) * F32 for _, s in shapes)
    O = optimizer_state_bytes(optimizer, [s for _, s in shapes])
    comp = {"W_bytes": W, "G_bytes": W, "O_bytes": O}
    baseline = W + W + O
    sim.phase = "ph_zero_grad"
    for p in params:
        sim.drop(p.grad)
        p.grad = None
    for _ in range(batch // microbatch):
        sim.phase = "ph_forward"
        xc = sim.new((microbatch, n_tokens, token_dim), tag="input_gather")
        yc = sim.new((microbatch,), I64, "target_gather")
        logits = net_forward(sim, net, xc)
        loss = sim.cross_entropy_sum_div(logits, yc, batch)
        sim.phase = "ph_backward"
        seed = sim.new((), tag="ones_like_seed")
        sim.run_engine(loss, seed)
        sim.drop(seed)
        sim.drop_all([xc, yc, loss])
    sim.phase = "ph_clip"
    clip_events(sim, len(params))
    sim.phase = "ph_step"
    pshapes = [s for _, s in shapes]
    if optimizer == "adam":
        adam_step(sim, pshapes)
    elif optimizer == "sgd_momentum":
        sgd_momentum_step(sim, pshapes)
    elif optimizer == "adafactor":
        adafactor_step(sim, pshapes)
    elif optimizer != "sgd":
        raise ValueError(f"unknown optimizer {optimizer!r}")
    if sum(sim.ev) != 0:
        raise AssertionError(f"simulator closure error {sum(sim.ev)} bytes")
    if any(p.grad is None for p in params):
        raise AssertionError("simulator: a parameter has no gradient after the update")
    return sim, baseline, comp


def reconstruct(sim, baseline):
    """Profiler rule: peak = baseline + max(0, max prefix); per-phase peaks as lb.profiler.reconstruct."""
    cum, peak, peak_i = 0, 0, None
    phase_peak = {}
    last = {}
    for i, (nb, (ph, _)) in enumerate(zip(sim.ev, sim.tags)):
        if ph not in phase_peak:
            phase_peak[ph] = baseline + cum
        cum += nb
        phase_peak[ph] = max(phase_peak[ph], baseline + cum)
        if cum > peak:
            peak, peak_i = cum, i
        last[ph] = i
    peak_phase = "start (baseline)" if peak_i is None else sim.tags[peak_i][0]
    return {"baseline_bytes": baseline, "peak_bytes": baseline + peak, "peak_above_baseline": peak,
            "peak_phase": peak_phase, "phase_peak": phase_peak, "n_events": len(sim.ev)}


def static_saved_bytes(dims, structure, d, T, microbatch, checkpointing, batch=64):
    """A_static of baseline B3: one microbatch's inputs and targets plus every non-parameter storage that autograd
    saves in that microbatch's forward (node saved tensors; checkpoint frames' saved block inputs; the reversible
    stack's saved output), once per storage, no lifetimes, no transients. Checkpoint RNG states are excluded."""
    if checkpointing and structure in REVERSIBLE:
        raise ValueError("checkpointing is not combined with reversible structures")
    n_tokens, token_dim, classes = dims
    sim = Sim()
    net, _ = build_net(sim, dims, structure, d, T, checkpointing)
    param_bufs = {id(p.buf) for p in net["params"]}
    xc = sim.new((microbatch, n_tokens, token_dim), tag="input_gather")
    yc = sim.new((microbatch,), I64, "target_gather")
    loss = sim.cross_entropy_sum_div(net_forward(sim, net, xc), yc, batch)
    bufs = {id(xc.buf): xc.buf.nbytes, id(yc.buf): yc.buf.nbytes}
    seen, stack = {id(loss.grad_fn)}, [loss.grad_fn]
    while stack:
        n = stack.pop()
        for t in n.saved.values():
            if id(t.buf) not in param_bufs:
                bufs[id(t.buf)] = t.buf.nbytes
        if n.frame is not None:
            bufs[id(n.frame.inp.buf)] = n.frame.inp.buf.nbytes
        for e in n.edges:
            if e is not None and id(e[0]) not in seen:
                seen.add(id(e[0]))
                stack.append(e[0])
    return sum(bufs.values())
