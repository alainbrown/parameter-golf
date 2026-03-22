"""FP8 training for Parameter Golf. Built from verified H100 test components."""

from __future__ import annotations
import copy, glob, io, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
try:
    import zstandard; _COMP = "zstd"
except ImportError:
    _COMP = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Config ---
DATA = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
TOK = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
RUN_ID = os.environ.get("RUN_ID", str(uuid.uuid4()))
SEED = int(os.environ.get("SEED", 1337))
V = int(os.environ.get("VOCAB_SIZE", 1024))
SEQ = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
BATCH = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
LAYERS = int(os.environ.get("NUM_LAYERS", 6))
DIM = int(os.environ.get("MODEL_DIM", 512))
NH = int(os.environ.get("NUM_HEADS", 8))
NKV = int(os.environ.get("NUM_KV_HEADS", 4))
MLP_M = int(os.environ.get("MLP_MULT", 3))
RECUR = int(os.environ.get("RECURRENCE_LOOPS", 2))
ITERS = int(os.environ.get("ITERATIONS", 20000))
WARMDOWN = int(os.environ.get("WARMDOWN_ITERS", 3000))
WARMUP = int(os.environ.get("WARMUP_STEPS", 20))
MAX_S = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
VAL_EVERY = int(os.environ.get("VAL_LOSS_EVERY", 1000))
LOG_EVERY = int(os.environ.get("TRAIN_LOG_EVERY", 200))
VAL_BS = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", 64))
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", 16))
SOFTCAP = 30.0
QK_GAIN = float(os.environ.get("QK_GAIN_INIT", 1.5))
EMBED_STD = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
MAT_LR = float(os.environ.get("MATRIX_LR", 0.02))
SC_LR = float(os.environ.get("SCALAR_LR", 0.02))
TIED_LR = float(os.environ.get("TIED_EMBED_LR", 0.03))
MU_MOM = float(os.environ.get("MUON_MOMENTUM", 0.99))
MU_STEPS = int(os.environ.get("MUON_BACKEND_STEPS", 5))
MU_WU_S = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
MU_WU_N = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
CLIP = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
B1, B2, EPS = 0.9, 0.95, 1e-8
N_VB = int(os.environ.get("NUM_VALUE_BANKS", 5))
BG_BK = int(os.environ.get("BIGRAM_BUCKETS", 4096))
BG_DIM = int(os.environ.get("BIGRAM_DIM", 128))
SWA_EVERY = int(os.environ.get("SWA_EVERY", 0))
SWA_FRAC = float(os.environ.get("SWA_START_FRAC", 0.4))
DO_VE = bool(int(os.environ.get("VE", "1")))
DO_SMEAR = bool(int(os.environ.get("SMEAR", "0")))
DO_BIGRAM = bool(int(os.environ.get("BIGRAM", "1")))
DO_QAT = bool(int(os.environ.get("QAT", "0")))

# --- FP8 ops (tensor scales, verified on H100) ---
_XS = torch.tensor(100.0 / 448.0)
_WS = torch.tensor(1.6 / 448.0)
_GS = torch.tensor(1.0)

@torch.library.custom_op("pg::fwd", mutates_args=())
def _fwd(x: Tensor, w: Tensor, xs: Tensor, ws: Tensor, gs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x, w):
        x8 = x.div(xs).to(torch.float8_e4m3fn)
        w8 = w.div(ws).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(x8, w8.T.contiguous().T, out_dtype=torch.bfloat16,
                               scale_a=xs.float().to(x.device), scale_b=ws.float().to(x.device), use_fast_accum=True)
        return out, x8, w8
    return impl(x, w)

@_fwd.register_fake
def _(x, w, xs, ws, gs):
    return x @ w, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("pg::bwd", mutates_args=())
def _bwd(g: Tensor, x8: Tensor, w8: Tensor, xs: Tensor, ws: Tensor, gs: Tensor) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(g, x8, w8):
        g8 = g.div(gs).to(torch.float8_e5m2)
        dx = torch._scaled_mm(g8, w8.T, out_dtype=torch.bfloat16, scale_a=gs.float().to(g.device), scale_b=ws.float().to(g.device), use_fast_accum=False)
        dw = torch._scaled_mm(x8.T.contiguous(), g8.T.contiguous().T, out_dtype=torch.float32, scale_a=xs.float().to(g.device), scale_b=gs.float().to(g.device), use_fast_accum=False)
        return dx, dw
    return impl(g, x8, w8)

@_bwd.register_fake
def _(g, x8, w8, xs, ws, gs):
    return x8.to(torch.bfloat16), w8.to(torch.float32)

def _bfn(ctx, go, *_):
    x8, w8, xs, ws, gs = ctx.saved_tensors
    return (*torch.ops.pg.bwd(go, x8, w8, xs, ws, gs), None, None, None)
def _cfn(ctx, inputs, output):
    _, _, xs, ws, gs = inputs; _, x8, w8 = output
    ctx.save_for_backward(x8, w8, xs, ws, gs); ctx.set_materialize_grads(False)
_fwd.register_autograd(_bfn, setup_context=_cfn)

# --- QAT ---
def _fq(w):
    wf = w.float(); amax = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    sc = amax / 31.0; q = (wf / sc).round().clamp(-32, 31)
    return w + ((q * sc).to(w.dtype) - w).detach()

# --- Layers ---
CTRL = ("attn_scale", "mlp_scale", "resid_mix", "q_gain")

class FP8Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        w = torch.empty(d_in, d_out); nn.init.orthogonal_(w, gain=1.0)
        self.weight = nn.Parameter(w.bfloat16())
    def forward(self, x):
        w = _fq(self.weight) if DO_QAT and self.training else self.weight
        if self.training:
            flat = x.reshape(-1, self.d_in); n = flat.shape[0]
            pad = (16 - n % 16) % 16
            if pad: flat = F.pad(flat, (0, 0, 0, pad))
            out = torch.ops.pg.fwd(flat, w, _XS, _WS, _GS)[0]
            if pad: out = out[:n]
            return out.reshape(*x.shape[:-1], self.d_out)
        return x @ w.to(x.dtype)

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), 3.0, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(x.dtype))[None, None, :]
        return (1 - g) * x + g * F.pad(x[:, :-1], (0, 0, 1, 0))

class BigramHash(nn.Module):
    def __init__(self, vocab, buckets, dim, out_dim):
        super().__init__()
        self.buckets = buckets
        self.embed = nn.Embedding(buckets, dim); nn.init.normal_(self.embed.weight, std=0.02)
        self.proj = FP8Linear(dim, out_dim) if dim != out_dim else None
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def forward(self, ids):
        t = ids.to(torch.int32); h = torch.zeros_like(t)
        h[:, 1:] = torch.bitwise_xor(36313 * t[:, 1:], 27191 * t[:, :-1]) % (self.buckets - 1)
        e = self.embed(h.long())
        if self.proj is not None: e = self.proj(e)
        return e * self.scale.to(e.dtype)

class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), persistent=False)
        self._c = self._s = None; self._len = 0
    def forward(self, n, dev, dt):
        if self._c is None or self._len != n or self._c.device != dev:
            f = torch.outer(torch.arange(n, device=dev, dtype=self.inv_freq.dtype), self.inv_freq.to(dev))
            self._c, self._s, self._len = f.cos()[None, None], f.sin()[None, None], n
        return self._c.to(dt), self._s.to(dt)

def rope(x, c, s):
    h = x.size(-1) // 2
    return torch.cat((x[..., :h]*c + x[..., h:]*s, -x[..., :h]*s + x[..., h:]*c), -1)

class Attn(nn.Module):
    def __init__(self, dim, nh, nkv, rb, qkg):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, dim // nh
        self.q = FP8Linear(dim, dim); self.k = FP8Linear(dim, nkv * self.hd)
        self.v = FP8Linear(dim, nkv * self.hd); self.proj = FP8Linear(dim, dim)
        self.qg = nn.Parameter(torch.full((nh,), qkg, dtype=torch.float32))
        self.rot = Rotary(self.hd, rb)
    def forward(self, x, ve=None):
        B, T, D = x.shape
        q = self.q(x).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        if ve is not None: v = v + ve.reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = F.rms_norm(q, (self.hd,)), F.rms_norm(k, (self.hd,))
        c, s = self.rot(T, x.device, q.dtype); q, k = rope(q, c, s), rope(k, c, s)
        q = q * self.qg.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.nkv != self.nh))
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, D))

class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.fc = FP8Linear(dim, dim * mult); self.proj = FP8Linear(dim * mult, dim)
    def forward(self, x): return self.proj(torch.relu(self.fc(x)).square())

class Block(nn.Module):
    def __init__(self, dim, nh, nkv, mult, rb, qkg):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(), RMSNorm()
        self.attn = Attn(dim, nh, nkv, rb, qkg); self.mlp = MLP(dim, mult)
        self.as_ = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ms = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
    def forward(self, x, x0, ve=None):
        m = self.mix.to(x.dtype)
        x = m[0][None, None, :] * x + m[1][None, None, :] * x0
        x = x + self.as_.to(x.dtype)[None, None, :] * self.attn(self.ln1(x), ve)
        x = x + self.ms.to(x.dtype)[None, None, :] * self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.recur, self.n_vb = RECUR, N_VB
        self.emb = nn.Embedding(V, DIM); nn.init.normal_(self.emb.weight, std=EMBED_STD)
        self.smear = SmearGate(DIM) if DO_SMEAR else None
        self.bigram = BigramHash(V, BG_BK, BG_DIM, DIM) if DO_BIGRAM else None
        self.blocks = nn.ModuleList([Block(DIM, NH, NKV, MLP_M, 10000.0, QK_GAIN) for _ in range(LAYERS)])
        self.norm = RMSNorm()
        kv_dim = NKV * (DIM // NH)
        self.ve = nn.Parameter(0.01 * torch.randn(N_VB, V, kv_dim, dtype=torch.bfloat16)) if DO_VE else None
        eff = LAYERS * RECUR
        for b in self.blocks:
            b.attn.proj.weight.data.mul_(1.0 / math.sqrt(2 * eff))
            b.mlp.proj.weight.data.mul_(1.0 / math.sqrt(2 * eff))

    def forward(self, ids, tgt):
        x = self.emb(ids)
        if self.bigram is not None: x = x + self.bigram(ids)
        if self.smear is not None: x = self.smear(x)
        x = F.rms_norm(x, (DIM,)); x0 = x
        eff = 0
        for _ in range(self.recur):
            for i, block in enumerate(self.blocks):
                ve = self.ve[eff % self.n_vb][ids] if self.ve is not None else None
                x = block(x, x0, ve); eff += 1
        x = self.norm(x)
        lo = F.linear(x, self.emb.weight)
        lo = SOFTCAP * torch.tanh(lo / SOFTCAP)
        return F.cross_entropy(lo.reshape(-1, V).float(), tgt.reshape(-1))

    def logits(self, ids):
        x = self.emb(ids)
        if self.bigram is not None: x = x + self.bigram(ids)
        if self.smear is not None: x = self.smear(x)
        x = F.rms_norm(x, (DIM,)); x0 = x
        eff = 0
        for _ in range(self.recur):
            for i, block in enumerate(self.blocks):
                ve = self.ve[eff % self.n_vb][ids] if self.ve is not None else None
                x = block(x, x0, ve); eff += 1
        lo = F.linear(self.norm(x), self.emb.weight)
        return SOFTCAP * torch.tanh(lo / SOFTCAP)

# --- Muon ---
def newtonschulz(G, steps=10, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16(); X /= X.norm() + eps
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if G.size(0) > G.size(1) else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, mom, steps, nesterov=True):
        super().__init__(params, dict(lr=lr, mom=mom, steps=steps, nesterov=nesterov))
    @torch.no_grad()
    def step(self):
        ddp = dist.is_available() and dist.is_initialized()
        ws = dist.get_world_size() if ddp else 1; rk = dist.get_rank() if ddp else 0
        for g in self.param_groups:
            ps = g["params"]; lr, mom, st, nest = g["lr"], g["mom"], g["steps"], g["nesterov"]
            n = sum(p.numel() for p in ps)
            flat = torch.zeros(n, device=ps[0].device, dtype=torch.bfloat16); c = 0
            for i, p in enumerate(ps):
                if i % ws == rk and p.grad is not None:
                    gr = p.grad; s = self.state[p]
                    if "buf" not in s: s["buf"] = torch.zeros_like(gr)
                    s["buf"].mul_(mom).add_(gr)
                    u = gr.add(s["buf"], alpha=mom) if nest else s["buf"]
                    u = newtonschulz(u, st); u *= max(1, u.size(0)/u.size(1))**0.5
                    flat[c:c+p.numel()] = u.reshape(-1)
                c += p.numel()
            if ddp: dist.all_reduce(flat)
            c = 0
            for p in ps:
                p.add_(flat[c:c+p.numel()].view_as(p).to(p.dtype), alpha=-lr); c += p.numel()

# --- Data ---
def load_shard(f):
    h = np.fromfile(f, dtype="<i4", count=256)
    return torch.from_numpy(np.fromfile(f, dtype="<u2", count=int(h[2]), offset=1024).astype(np.uint16, copy=False))

class Tokens:
    def __init__(self, pat):
        self.files = [Path(p) for p in sorted(glob.glob(pat))]
        assert self.files; self.fi = 0; self.tok = load_shard(self.files[0]); self.pos = 0
    def _next(self):
        self.fi = (self.fi + 1) % len(self.files); self.tok = load_shard(self.files[self.fi]); self.pos = 0
    def take(self, n):
        cs = []; r = n
        while r > 0:
            a = self.tok.numel() - self.pos
            if a <= 0: self._next(); continue
            k = min(r, a); cs.append(self.tok[self.pos:self.pos+k]); self.pos += k; r -= k
        return cs[0] if len(cs) == 1 else torch.cat(cs)

class Loader:
    def __init__(self, pat, rk, ws, dev):
        self.rk, self.ws, self.dev = rk, ws, dev; self.s = Tokens(pat)
    def batch(self, toks, sl, ga):
        lt = toks // (self.ws * ga); sp = lt + 1
        ch = self.s.take(sp * self.ws); st = self.rk * sp
        l = ch[st:st+sp].to(torch.int64)
        return l[:-1].reshape(-1, sl).to(self.dev, non_blocking=True), l[1:].reshape(-1, sl).to(self.dev, non_blocking=True)

# --- BPB ---
def build_luts(sp, vs, dev):
    n = max(int(sp.vocab_size()), vs)
    bb = np.zeros(n, dtype=np.int16); ls = np.zeros(n, dtype=np.bool_); bd = np.ones(n, dtype=np.bool_)
    for i in range(int(sp.vocab_size())):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i): continue
        bd[i] = False
        if sp.is_byte(i): bb[i] = 1; continue
        p = sp.id_to_piece(i)
        if p.startswith("\u2581"): ls[i] = True; p = p[1:]
        bb[i] = len(p.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=dev), torch.tensor(ls, dtype=torch.bool, device=dev), torch.tensor(bd, dtype=torch.bool, device=dev))

def load_val(pat, sl):
    fs = [Path(p) for p in sorted(glob.glob(pat))]
    t = torch.cat([load_shard(f) for f in fs]).contiguous()
    u = ((t.numel()-1)//sl)*sl; return t[:u+1]

def evaluate(model, rk, ws, dev, ga, vt, bl, ll, il):
    bs = VAL_BS // (ws * ga); ns = bs // SEQ
    ts = (vt.numel()-1)//SEQ; s0 = (ts*rk)//ws; s1 = (ts*(rk+1))//ws
    ls = torch.zeros((), device=dev, dtype=torch.float64)
    tc = torch.zeros((), device=dev, dtype=torch.float64)
    bc = torch.zeros((), device=dev, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for b in range(s0, s1, ns):
            be = min(b+ns, s1); r0 = b*SEQ; r1 = be*SEQ+1
            l = vt[r0:r1].to(device=dev, dtype=torch.int64, non_blocking=True)
            x, y = l[:-1].reshape(-1, SEQ), l[1:].reshape(-1, SEQ)
            loss = model(x, y).detach()
            n = float(y.numel()); ls += loss.to(torch.float64)*n; tc += n
            tb = bl[y.reshape(-1)].to(torch.int16)
            tb += (ll[y.reshape(-1)] & ~il[x.reshape(-1)]).to(torch.int16)
            bc += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (ls, tc, bc): dist.all_reduce(t)
    vl = ls/tc; bpt = vl.item()/math.log(2); tpb = tc.item()/bc.item()
    model.train(); return float(vl.item()), float(bpt*tpb)

def eval_sliding(base, rk, ws, dev, vt, bl, ll, il):
    stride, B = EVAL_STRIDE, EVAL_BATCH; n = vt.numel()-1
    wins = [w for w in range(0, n, stride) if min(w+SEQ, n)-w >= 1]
    my = wins[(len(wins)*rk)//ws:(len(wins)*(rk+1))//ws]
    ls = torch.zeros((), device=dev, dtype=torch.float64)
    tc = torch.zeros((), device=dev, dtype=torch.float64)
    bc = torch.zeros((), device=dev, dtype=torch.float64)
    base.eval()
    with torch.inference_mode():
        for bi in range(0, len(my), B):
            bw = my[bi:bi+B]; nb = len(bw)
            x = torch.zeros(nb, SEQ, dtype=torch.int64, device=dev)
            y = torch.zeros(nb, SEQ, dtype=torch.int64, device=dev); wlens = []
            for j, w in enumerate(bw):
                end = min(w+SEQ, n); wl = end-w
                ch = vt[w:end+1].to(torch.int64)
                x[j,:wl] = ch[:-1]; y[j,:wl] = ch[1:]; wlens.append(wl)
            lo = base.logits(x)
            nll = F.cross_entropy(lo.reshape(-1, lo.size(-1)).float(), y.reshape(-1), reduction='none').reshape(nb, SEQ)
            for j, w in enumerate(bw):
                wl = wlens[j]; s = 0 if w == 0 else max(wl-stride, 0)
                ls += nll[j,s:wl].to(torch.float64).sum(); tc += float(wl-s)
                tb = bl[y[j,s:wl]].to(torch.float64)
                tb += (ll[y[j,s:wl]] & ~il[x[j,s:wl]]).to(torch.float64); bc += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (ls, tc, bc): dist.all_reduce(t)
    vl = ls/tc; bpt = vl.item()/math.log(2); tpb = tc.item()/bc.item()
    base.train(); return float(vl.item()), float(bpt*tpb)

# --- Quantization ---
INT6 = 31; FP16_PASS = ("emb.weight",)

def quantize(sd):
    Q, S, D_, P, PD, QM = {}, {}, {}, {}, {}, {}
    st = dict.fromkeys(("pc","nt","nf","nn","bt","ip"), 0); CQ = 99.99984/100
    for nm, t in sd.items():
        t = t.detach().cpu().contiguous()
        st["pc"] += t.numel(); st["nt"] += 1; st["bt"] += t.numel()*t.element_size()
        if not t.is_floating_point(): st["nn"] += 1; P[nm] = t; st["ip"] += t.numel()*t.element_size(); continue
        if t.numel() <= 65536 or any(fp in nm for fp in FP16_PASS):
            if any(c in nm for c in CTRL): kept = t.float().contiguous()
            elif t.dtype in (torch.float32, torch.bfloat16): PD[nm] = str(t.dtype).removeprefix("torch."); kept = t.half().contiguous()
            else: kept = t
            P[nm] = kept; st["ip"] += kept.numel()*kept.element_size(); continue
        st["nf"] += 1; t32 = t.float()
        if t32.ndim == 2:
            ca = torch.quantile(t32.abs(), CQ, dim=1); sc = (ca/INT6).clamp_min(1.0/INT6)
            q = torch.clamp(torch.round(torch.clamp(t32, -ca[:,None], ca[:,None])/sc[:,None]), -(INT6+1), INT6).to(torch.int8).contiguous()
            QM[nm] = {"scheme":"per_row","axis":0}; Q[nm]=q; S[nm]=sc.half().contiguous()
        else:
            ca = float(torch.quantile(t32.abs().flatten(), CQ)); sc = torch.tensor(ca/INT6 if ca>0 else 1.0)
            q = torch.clamp(torch.round(torch.clamp(t32,-ca,ca)/sc),-(INT6+1),INT6).to(torch.int8).contiguous()
            Q[nm]=q; S[nm]=sc
        D_[nm] = str(t.dtype).removeprefix("torch.")
        st["ip"] += Q[nm].numel()*Q[nm].element_size() + (S[nm].numel()*S[nm].element_size() if S[nm].ndim>0 else 4)
    obj = {"__quant_format__":"int6_v1","quantized":Q,"scales":S,"dtypes":D_,"passthrough":P}
    if QM: obj["qmeta"]=QM
    if PD: obj["passthrough_orig_dtypes"]=PD
    return obj, st

def dequantize(obj):
    out = {}; qm = obj.get("qmeta",{}); pd = obj.get("passthrough_orig_dtypes",{})
    for nm, q in obj["quantized"].items():
        dt = getattr(torch, obj["dtypes"][nm]); s = obj["scales"][nm]
        if qm.get(nm,{}).get("scheme")=="per_row" or s.ndim>0:
            out[nm] = (q.float()*s.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(dt).contiguous()
        else: out[nm] = (q.float()*float(s.item())).to(dt).contiguous()
    for nm, t in obj["passthrough"].items():
        o = t.detach().cpu().contiguous(); d = pd.get(nm)
        if d: o = o.to(getattr(torch, d)).contiguous()
        out[nm] = o
    return out

# --- Main ---
def main():
    global _GS, newtonschulz
    code = Path(__file__).read_text(encoding="utf-8")
    newtonschulz = torch.compile(newtonschulz)

    ddp = "RANK" in os.environ
    rk = int(os.environ.get("RANK", 0)); ws = int(os.environ.get("WORLD_SIZE", 1))
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    ga = 8 // ws; gs = 1.0 / ga
    dev = torch.device("cuda", lr_); torch.cuda.set_device(dev)
    if ddp: dist.init_process_group("nccl", device_id=dev); dist.barrier()
    master = rk == 0
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_cudnn_sdp, enable_mem_efficient_sdp, enable_math_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    lf = None
    if master: os.makedirs("logs", exist_ok=True); lf = f"logs/{RUN_ID}.txt"; print(lf)
    def log(m, con=True):
        if not master: return
        if con: print(m)
        if lf:
            with open(lf, "a") as f: print(m, file=f)
    log(code, con=False); log("="*100, con=False)
    log(f"Python {sys.version}", con=False); log(f"PyTorch {torch.__version__}", con=False)
    log(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout, con=False)
    log("="*100, con=False)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    sp = spm.SentencePieceProcessor(model_file=TOK)
    vt = load_val(os.path.join(DATA, "fineweb_val_*.bin"), SEQ)
    bl, ll, il = build_luts(sp, V, dev)

    _GS = torch.tensor(gs * 0.75 / 448.0)
    flags = []
    if DO_VE: flags.append("VE")
    if DO_SMEAR: flags.append("SMEAR")
    if DO_BIGRAM: flags.append("BIGRAM")
    if DO_QAT: flags.append("QAT")
    if SWA_EVERY > 0: flags.append(f"SWA={SWA_EVERY}")
    if RECUR > 1: flags.append(f"RECUR={RECUR}")
    log(f"features: {','.join(flags) if flags else 'baseline'}")
    log(f"fp8: xs={_XS:.4f} ws={_WS:.6f} gs={_GS:.2e}")

    base = GPT().to(dev).bfloat16()
    with torch.no_grad():
        for nm, p in base.named_parameters():
            if p.ndim < 2 or any(c in nm for c in CTRL):
                if p.dtype != torch.float32: p.data = p.data.float()

    model = torch.compile(base, fullgraph=True)
    model = DDP(model, device_ids=[lr_], broadcast_buffers=False) if ddp else model

    bp = list(base.blocks.named_parameters())
    mat_p = [p for n, p in bp if p.ndim == 2 and not any(c in n for c in CTRL)]
    sc_p = [p for n, p in bp if p.ndim < 2 or any(c in n for c in CTRL)]
    opt_e = torch.optim.Adam([{"params": [base.emb.weight], "lr": TIED_LR, "base_lr": TIED_LR}], betas=(B1, B2), eps=EPS, fused=True)
    if base.ve is not None: opt_e.param_groups[0]["params"].append(base.ve)
    opt_m = Muon(mat_p, lr=MAT_LR, mom=MU_MOM, steps=MU_STEPS)
    for g in opt_m.param_groups: g["base_lr"] = MAT_LR
    opt_s = torch.optim.Adam([{"params": sc_p, "lr": SC_LR, "base_lr": SC_LR}], betas=(B1, B2), eps=EPS, fused=True)
    opts = [opt_e, opt_m, opt_s]

    eff = LAYERS * RECUR
    log(f"params:{sum(p.numel() for p in base.parameters())} ws:{ws} ga:{ga} layers:{LAYERS}x{RECUR}={eff}")
    log(f"lr:embed={TIED_LR} mat={MAT_LR} scalar={SC_LR}")
    log(f"batch:{BATCH} seq:{SEQ} iters:{ITERS} warmup:{WARMUP} warmdown:{WARMDOWN} max_s:{MAX_S}")

    loader = Loader(os.path.join(DATA, "fineweb_train_*.bin"), rk, ws, dev)
    def zg():
        for o in opts: o.zero_grad(set_to_none=True)
    mms = 1000*MAX_S if MAX_S > 0 else None

    def lr_scale(step, ems):
        if WARMDOWN <= 0: return 1.0
        if mms is None:
            ws_ = max(ITERS - WARMDOWN, 0)
            return max((ITERS-step)/max(WARMDOWN,1), 0.0) if ws_ <= step < ITERS else 1.0
        sms = ems/max(step,1); wms = WARMDOWN*sms; rem = max(mms-ems, 0.0)
        return rem/max(wms, 1e-9) if rem <= wms else 1.0

    if WARMUP > 0:
        ms0 = {n: t.detach().cpu().clone() for n, t in base.state_dict().items()}
        os0 = [copy.deepcopy(o.state_dict()) for o in opts]
        model.train()
        for ws_ in range(WARMUP):
            zg()
            for mi in range(ga):
                if ddp: model.require_backward_grad_sync = mi == ga-1
                x, y = loader.batch(BATCH, SEQ, ga)
                loss = model(x, y); (loss*gs).backward()
            for o in opts: o.step(); zg()
            if WARMUP <= 20 or (ws_+1)%10==0 or ws_+1==WARMUP: log(f"warmup:{ws_+1}/{WARMUP}")
        base.load_state_dict(ms0, strict=True)
        for o, s in zip(opts, os0, strict=True): o.load_state_dict(s)
        zg()
        if ddp: model.require_backward_grad_sync = True
        loader = Loader(os.path.join(DATA, "fineweb_train_*.bin"), rk, ws, dev)

    swa_snaps = []; tms = 0.0; stop = None; _pcms = 0.0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step = 0
    while True:
        last = step == ITERS or (stop is not None and step >= stop)
        if last or (VAL_EVERY > 0 and step % VAL_EVERY == 0):
            torch.cuda.synchronize(); tms += 1000*(time.perf_counter()-t0)
            vl, vb = evaluate(model, rk, ws, dev, ga, vt, bl, ll, il)
            log(f"step:{step}/{ITERS} val_loss:{vl:.4f} val_bpb:{vb:.4f} time:{tms:.0f}ms avg:{tms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last:
            if stop is not None and step < ITERS: log(f"stopping_early: time:{tms:.0f}ms step:{step}/{ITERS}")
            break

        sc = lr_scale(step, tms + 1000*(time.perf_counter()-t0))
        zg(); tl = torch.zeros((), device=dev)
        for mi in range(ga):
            if ddp: model.require_backward_grad_sync = mi == ga-1
            x, y = loader.batch(BATCH, SEQ, ga)
            loss = model(x, y); tl += loss.detach(); (loss*gs).backward()
        tl /= ga

        frac = min(step/MU_WU_N, 1.0) if MU_WU_N > 0 else 1.0
        for g in opt_m.param_groups: g["mom"] = (1-frac)*MU_WU_S + frac*MU_MOM
        for o in opts:
            for g in o.param_groups: g["lr"] = g["base_lr"]*sc
        if CLIP > 0: torch.nn.utils.clip_grad_norm_(base.parameters(), CLIP)
        for o in opts: o.step(); zg()

        if SWA_EVERY > 0 and sc < 1.0 and sc <= SWA_FRAC and step % SWA_EVERY == 0:
            swa_snaps.append({n: t.detach().cpu().clone() for n, t in base.state_dict().items()})

        step += 1; ams = tms + 1000*(time.perf_counter()-t0)
        if step == 3: _pcms = ams
        if LOG_EVERY > 0 and (step <= 10 or step % LOG_EVERY == 0 or stop is not None):
            pa = (ams-_pcms)/(step-2) if step > 2 else 0
            log(f"step:{step}/{ITERS} loss:{tl.item():.4f} time:{ams:.0f}ms avg:{ams/step:.2f}ms pcavg:{pa:.2f}ms lr_scale:{sc:.4f}")
            if step == 1 or step % 200 == 0:
                wa = [m.weight.abs().amax().item() for m in base.modules() if isinstance(m, FP8Linear)]
                if wa: log(f"  fp8_amax: w_min={min(wa):.4f} w_max={max(wa):.4f} w_med={sorted(wa)[len(wa)//2]:.4f}")
                if swa_snaps: log(f"  swa_snaps: {len(swa_snaps)}")
        hit = mms is not None and ams >= mms
        if ddp and mms is not None:
            ht = torch.tensor(int(hit), device=dev); dist.all_reduce(ht, op=dist.ReduceOp.MAX); hit = bool(ht.item())
        if stop is None and hit: stop = step

    log(f"peak_mem:{torch.cuda.max_memory_allocated()//1024//1024}MiB")

    if swa_snaps:
        log(f"swa: averaging {len(swa_snaps)} snapshots")
        avg = {}
        for key in swa_snaps[0]:
            ts = [s[key] for s in swa_snaps]
            avg[key] = torch.stack(ts).float().mean(0).to(ts[0].dtype) if ts[0].is_floating_point() else ts[-1]
        base.load_state_dict(avg); del swa_snaps

    torch.cuda.synchronize(); te = time.perf_counter()
    pql, pqb = evaluate(model, rk, ws, dev, ga, vt, bl, ll, il)
    torch.cuda.synchronize()
    log(f"pre_quant val_loss:{pql:.4f} val_bpb:{pqb:.4f} eval:{1000*(time.perf_counter()-te):.0f}ms")

    if master:
        torch.save(base.state_dict(), "final_model.pt")
        mb = os.path.getsize("final_model.pt"); cb = len(code.encode("utf-8"))
        log(f"model:{mb} code:{cb} total:{mb+cb}")

    qo, qs = quantize(base.state_dict())
    buf = io.BytesIO(); torch.save(qo, buf); raw = buf.getvalue()
    blob = zstandard.ZstdCompressor(level=22).compress(raw) if _COMP == "zstd" else zlib.compress(raw, 9)
    if master:
        with open("final_model.int8.ptz","wb") as f: f.write(blob)
        qb = os.path.getsize("final_model.int8.ptz"); cb = len(code.encode("utf-8"))
        log(f"int6+{_COMP}:{qb} code:{cb} total:{qb+cb} ratio:{qs['bt']/max(qs['ip'],1):.2f}x")

    if ddp: dist.barrier()
    with open("final_model.int8.ptz","rb") as f: disk = f.read()
    dec = zstandard.ZstdDecompressor().decompress(disk) if _COMP == "zstd" else zlib.decompress(disk)
    base.load_state_dict(dequantize(torch.load(io.BytesIO(dec), map_location="cpu")), strict=True)
    torch.cuda.synchronize(); te = time.perf_counter()
    ql, qb_ = eval_sliding(base, rk, ws, dev, vt, bl, ll, il)
    torch.cuda.synchronize()
    log(f"final_roundtrip val_loss:{ql:.4f} val_bpb:{qb_:.4f} eval:{1000*(time.perf_counter()-te):.0f}ms stride={EVAL_STRIDE}")
    log(f"final_roundtrip_exact val_loss:{ql:.8f} val_bpb:{qb_:.8f}")
    log(f"quant_gap: pre={pqb:.4f} post={qb_:.4f} gap={qb_-pqb:.4f}")
    if ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()
