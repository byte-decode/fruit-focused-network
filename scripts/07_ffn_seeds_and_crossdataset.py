"""
PILOT 10: Fruit-Focused Network (FFN) - error bars + real-photo cross-dataset test.

Same model and training as pilot 9. Pick the tasks for this session in
CFG["tasks"] (summary.txt is rewritten after every training, so a partial
run is still usable):

  "seeds"     FruitNet leave-one-fruit-out, extra seeds (default [1, 2]) for
              baseline / FFN / BS-only. Together with pilot 9 (seed 0) this
              gives 3 seeds per number.                        (~4-5 h)
  "crossdata" Train on ALL of FruitNet (each seed), then test on REAL photos
              of apple / banana / orange from the Fresh-Rotten dataset:
              different people, cameras and backgrounds, no pasting at all.
              Answers "did FFN only learn the pasted look?"     (~1.5-2 h)
  "fr_lofo"   The same comparison on Fresh-Rotten's own leave-one-fruit-out
              (second dataset).                                 (~1 h)

Suggested split over two Kaggle sessions:
  session 1: tasks=["crossdata", "fr_lofo"]
  session 2: tasks=["seeds"]

Kaggle inputs (add BOTH for crossdata / fr_lofo; FruitNet alone is enough for seeds):
  shashwatwork/fruitnet-indian-fruits-dataset-with-quality
  sriramr/fruits-fresh-and-rotten-for-classification
Internet ON, GPU T4.   Run:  !python -u pilot10_ffn.py
Outputs: /kaggle/working/pilot10/  summary.txt, results.json, progress.txt
"""
print("pilot10: starting, importing libraries ...", flush=True)
import os, json, random, hashlib, time, copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, recall_score
from PIL import Image

CFG = dict(
    input_root="/kaggle/input",
    tasks=["crossdata", "fr_lofo"],   # any of "seeds", "crossdata", "fr_lofo"
    seeds_lofo=[1, 2],           # seeds for the "seeds" task (seed 0 = pilot 9)
    seeds_cross=[0, 1, 2],       # seeds for "crossdata"
    seeds_fr=[0],                # seeds for "fr_lofo"
    cache_size=256, img=224,
    split_seed=42,
    dup_threshold=0.95,
    clipseg="CIDAS/clipseg-rd64-refined",
    mask_threshold=0.4,
    grow_px=12,                  # fruit mask enlarged by this much before cleaning backgrounds
    variants=["baseline", "FFN", "BS-only"],
    p_swap=0.8,                  # share of training images that get a swapped background
    att_weight=0.5,              # weight of the attention-map supervision loss
    bs=64, epochs=12, lr=3e-4, wd=1e-4, log_every=50,
    workers=2,
    out=None,
)
DATASETS = {
    "fruitnet": dict(path_filter="fruitnet", lofo_holdout_calib=True,
                     quality_keys={"Good": ["good"], "Bad": ["bad"], "Mixed": ["mixed"]},
                     fruits=["apple", "banana", "guava", "lime", "orange", "pomegranate"]),
    "freshrotten": dict(path_filter="fresh", lofo_holdout_calib=False,
                        quality_keys={"Good": ["fresh"], "Bad": ["rotten"]},
                        fruits=["apple", "banana", "orange"]),
}
DS = DATASETS["fruitnet"]
CFG["out"] = "/kaggle/working/pilot10"
os.makedirs(CFG["out"], exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_T0 = time.time()

CLASSES = ["Good", "Bad"]
QUALITY_KEYS = DS["quality_keys"]
FRUITS = DS["fruits"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def progress(i, total, what, every):
    if i == 0 or (i + 1) % every == 0 or i + 1 == total:
        log(f"  {what}: {i + 1}/{total}")


def log(msg):
    """Print with elapsed time, and also append to progress.txt."""
    line = f"[{(time.time() - _T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(os.path.join(CFG["out"], "progress.txt"), "a") as f:
        f.write(line + "\n")

CLASSES = ["Good", "Bad"]                         # label 0 / 1
QUALITY_KEYS = DS["quality_keys"]
FRUITS = DS["fruits"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def match_part(parts, keys):
    """Deepest path part that matches exactly one key (skips e.g. a top folder
    named 'Good, Bad & Mixed quality')."""
    for p in reversed(parts):
        hits = {k for k, words in keys.items() if any(w in p for w in words)}
        if len(hits) == 1:
            return hits.pop()
    return None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def index_images():
    rows, skipped = [], Counter()
    for dp, _, fs in os.walk(CFG["input_root"]):
        for fn in fs:
            if not fn.lower().endswith(IMG_EXT) or fn.startswith("._"):
                continue
            path = os.path.join(dp, fn)
            if DS["path_filter"] not in path.lower():
                continue
            parts = [p.lower() for p in os.path.relpath(path, CFG["input_root"]).split(os.sep)[:-1]]
            q = match_part(parts, QUALITY_KEYS)
            fr = match_part(parts, {f: [f] for f in FRUITS})
            if q is None or fr is None:
                skipped["/".join(parts[:3])] += 1
                continue
            rows.append(dict(path=path, quality=q, fruit=fr))
            if len(rows) % 2000 == 0:
                log(f"  found {len(rows)} images so far (now in {os.path.relpath(dp, CFG['input_root'])})")
    log(f"found {len(rows)} images; checking for exact duplicates")
    if skipped:
        print("Skipped (no quality or fruit in folder names):", dict(skipped))
    with ThreadPoolExecutor(8) as ex:
        hashes = []
        for i, h in enumerate(ex.map(md5, [r["path"] for r in rows])):
            hashes.append(h)
            progress(i, len(rows), "duplicate check", 2000)
    seen, uniq = set(), []
    for r, h in zip(rows, hashes):
        if h not in seen:
            seen.add(h); uniq.append(r)
    print(f"Found {len(rows)} images, {len(rows) - len(uniq)} exact duplicates removed -> {len(uniq)}")
    return sorted(uniq, key=lambda r: r["path"])


def load_cache(rows):
    S = CFG["cache_size"]
    def load(p):
        im = Image.open(p).convert("RGB")
        w, h = im.size; s = min(w, h)            # centre square crop, then resize
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return np.asarray(im.resize((S, S), Image.BILINEAR))
    print(f"Loading {len(rows)} images into memory ...")
    with ThreadPoolExecutor(8) as ex:
        X = np.empty((len(rows), S, S, 3), np.uint8)
        for i, a in enumerate(ex.map(load, [r["path"] for r in rows])):
            X[i] = a
            progress(i, len(rows), "images loaded", 1000)
    print(f"  cache {X.nbytes / 1e9:.2f} GB")
    return X


@torch.no_grad()
def near_duplicate_groups(Fe, thr, k=20):
    """Links each image to its (up to k) nearest neighbours with cosine >= thr,
    then joins linked images into groups (connected components). Fast and
    memory-safe even when many photos look alike."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(Fe)
    Fn = F.normalize(Fe, dim=1).to(DEV)
    nn_sim = torch.empty(n)
    rows_i, cols_i = [], []
    for s in range(0, n, 1024):
        sim = Fn[s:s + 1024] @ Fn.T
        b = sim.shape[0]
        sim[torch.arange(b), torch.arange(s, s + b)] = -1
        vals, idx = sim.topk(min(k, n - 1), dim=1)
        nn_sim[s:s + b] = vals[:, 0].cpu()
        keep = vals >= thr
        r = torch.arange(s, s + b, device=sim.device).unsqueeze(1).expand_as(idx)
        rows_i.append(r[keep].cpu()); cols_i.append(idx[keep].cpu())
        progress(s // 1024, (n + 1023) // 1024, "duplicate-scan chunks", 5)
    r, c = torch.cat(rows_i).numpy(), torch.cat(cols_i).numpy()
    log(f"near-duplicate links: {len(r)}")
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, groups = connected_components(graph, directed=False)
    return groups, nn_sim.numpy()


def group_split(idx, Y, fruit_arr, groups, frac, seed):
    """Split idx into (rest, held) with ~frac in held, groups never shared."""
    k = max(2, int(round(1 / frac)))
    strat = np.array([f"{fruit_arr[i]}_{Y[i]}" for i in idx])
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    rest, held = next(sgk.split(idx, strat, groups[idx]))
    return idx[rest], idx[held]


def make_splits(Y, fruit_arr, groups):
    s = CFG["split_seed"]
    all_idx = np.arange(len(Y))
    strat = [f"{fruit_arr[i]}_{Y[i]}" for i in all_idx]
    # standard, random (leaky if near-duplicates exist)
    tr, tmp = train_test_split(all_idx, test_size=0.30, stratify=strat, random_state=s)
    va, te = train_test_split(tmp, test_size=0.50, stratify=[strat[i] for i in tmp], random_state=s)
    splits = {"random": dict(tr=tr, va=va, te=te)}
    # standard, group-aware (honest)
    rest, te = group_split(all_idx, Y, fruit_arr, groups, 0.15, s)
    tr, va = group_split(rest, Y, fruit_arr, groups, 0.15, s)
    splits["group"] = dict(tr=tr, va=va, te=te)
    # leave-one-fruit-out
    order = FRUITS[:]; random.Random(s).shuffle(order)
    lofo = []
    for i, test_f in enumerate(order):
        calib_f = order[(i + 1) % len(order)]
        train_f = [f for f in order if f not in ((test_f, calib_f) if DS["lofo_holdout_calib"] else (test_f,))]
        tr_all = np.where(np.isin(fruit_arr, train_f))[0]
        tr, va = group_split(tr_all, Y, fruit_arr, groups, 0.15, s)
        lofo.append(dict(fold=i, test=test_f, calib=calib_f, train=train_f, tr=tr, va=va,
                         ca=np.where(fruit_arr == calib_f)[0], te=np.where(fruit_arr == test_f)[0]))
    return splits, lofo






# ---------------------------------------------------------- segmentation
@torch.no_grad()
def segment(X, fruit_arr):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    proc = CLIPSegProcessor.from_pretrained(CFG["clipseg"])
    model = CLIPSegForImageSegmentation.from_pretrained(CFG["clipseg"]).eval().to(DEV)
    log("CLIPSeg loaded")
    S = X.shape[1]
    M = np.zeros((len(X), S, S), np.uint8)
    probs_max = np.zeros(len(X), np.float32)
    bs = 32
    n_b = (len(X) + bs - 1) // bs
    for b in range(n_b):
        sl = slice(b * bs, (b + 1) * bs)
        imgs = [Image.fromarray(a) for a in X[sl]]
        texts = [("an " if f[0] in "aeiou" else "a ") + f for f in fruit_arr[sl]]
        inp = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to(DEV)
        logits = model(**inp).logits
        if logits.dim() == 2:
            logits = logits[None]
        p = torch.sigmoid(F.interpolate(logits[:, None].float(), size=(S, S), mode="bilinear",
                                        align_corners=False))[:, 0].cpu()
        for j in range(len(p)):
            pm = p[j].max().item()
            m = p[j] >= CFG["mask_threshold"]
            if m.float().mean() < 0.01:          # nothing confident: take the top region
                m = p[j] >= 0.5 * pm
            M[b * bs + j] = m.numpy()
            probs_max[b * bs + j] = pm
        progress(b, n_b, "CLIPSeg batches", 40)
    del model
    torch.cuda.empty_cache()
    return M, probs_max


def grow(M, px):
    out = np.empty_like(M)
    k = 2 * px + 1
    for s in range(0, len(M), 512):
        t = torch.from_numpy(M[s:s + 512]).float()[:, None]
        out[s:s + 512] = (F.max_pool2d(t, k, stride=1, padding=px)[:, 0] > 0).numpy()
    return out



# ------------------------------------------------------ clean backgrounds
@torch.no_grad()
def clean_backgrounds(X, Mg):
    """Remove the fruit from every photo: the (enlarged) fruit area is filled
    with a smooth average of the surrounding background."""
    Bg = np.empty_like(X)
    n = len(X)
    for s in range(0, n, 128):
        x = torch.from_numpy(X[s:s + 128]).to(DEV).permute(0, 3, 1, 2).float()
        keep = 1 - torch.from_numpy(Mg[s:s + 128]).to(DEV).float()[:, None]
        num = F.avg_pool2d(x * keep, 63, 1, 31, count_include_pad=False)
        den = F.avg_pool2d(keep, 63, 1, 31, count_include_pad=False)
        glob = (x * keep).sum((2, 3), keepdim=True) / keep.sum((2, 3), keepdim=True).clamp(min=1)
        grey = torch.tensor(MEAN, device=DEV).view(1, 3, 1, 1) * 255
        glob = torch.where(keep.sum((2, 3), keepdim=True) > 50, glob, grey)
        fill = torch.where(den > 0.05, num / den.clamp(min=1e-6), glob)
        out = x * keep + fill * (1 - keep)
        Bg[s:s + 128] = out.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        progress(s // 128, (n + 127) // 128, "cleaning backgrounds", 40)
    return Bg


# ----------------------------------------------------------------- data
def to_t(a):
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255)


def composite(D, i, j):
    """Fruit of photo i on the cleaned background of photo j."""
    X, M, Bg, Y = D
    m = torch.from_numpy(M[i]).float()[None]
    return to_t(X[i]) * m + to_t(Bg[j]) * (1 - m)


JITTER = T.ColorJitter(0.2, 0.2)


class FruitData(Dataset):
    """train=True : random crop/flip (applied to photo AND mask), colour jitter,
                    background swap with probability p_swap from swap_pool.
       pair       : fixed partner per item for the swapped test (or None)."""
    def __init__(self, D, idx, train=False, swap_pool=None, pair=None):
        self.D, self.idx, self.train = D, np.asarray(idx), train
        self.swap_pool, self.pair = swap_pool, pair
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        X, M, Bg, Y = self.D
        i = self.idx[k]
        if self.pair is not None:
            x = composite(self.D, i, self.pair[k])
        elif self.train and self.swap_pool is not None and random.random() < CFG["p_swap"]:
            x = composite(self.D, i, int(random.choice(self.swap_pool)))
        else:
            x = to_t(X[i])
        m = torch.from_numpy(M[i]).float()[None]
        if self.train:
            t, l, h, w = T.RandomResizedCrop.get_params(x, scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3))
            x = TF.resized_crop(x, t, l, h, w, [CFG["img"], CFG["img"]], antialias=True)
            m = TF.resized_crop(m, t, l, h, w, [CFG["img"], CFG["img"]], antialias=True)
            if random.random() < 0.5:
                x, m = TF.hflip(x), TF.hflip(m)
            x = JITTER(x)
        else:
            x = TF.resize(x, [CFG["img"], CFG["img"]], antialias=True)
            m = TF.resize(m, [CFG["img"], CFG["img"]], antialias=True)
        m7 = F.adaptive_avg_pool2d(m, 7)                     # target for the attention map
        return TF.normalize(x, MEAN, STD), int(Y[i]), m7


def loader(D, idx, train=False, swap_pool=None, pair=None, bs=None):
    return DataLoader(FruitData(D, idx, train, swap_pool, pair), batch_size=bs or CFG["bs"],
                      shuffle=train, num_workers=CFG["workers"], pin_memory=True, drop_last=train)


def opposite_pairs(Y, idx, seed=0):
    """For each test photo, a test photo of the opposite class (its background is used)."""
    rng = np.random.RandomState(seed)
    idx = np.asarray(idx)
    out = []
    for i in idx:
        cand = idx[Y[idx] != Y[i]]
        out.append(int(rng.choice(cand)) if len(cand) else int(rng.choice(idx)))
    return np.array(out)


# ---------------------------------------------------------------- model
class FruitFocusedNet(nn.Module):
    """MobileNetV3-Large. attention=False -> plain average pooling (baseline)."""
    def __init__(self, attention):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.features = base.features                          # 960 x 7 x 7
        self.att = nn.Conv2d(960, 1, 1) if attention else None
        if self.att is not None:
            nn.init.zeros_(self.att.weight); nn.init.zeros_(self.att.bias)   # starts as average pooling
        self.head = nn.Sequential(base.classifier[0], nn.Hardswish(), nn.Dropout(0.2), nn.Linear(1280, 2))

    def forward(self, x, return_map=False):
        f = self.features(x)
        if self.att is not None:
            a_logit = self.att(f)
            a = torch.sigmoid(a_logit)
            pooled = (f * a).sum((2, 3)) / a.sum((2, 3)).clamp(min=1e-4)
        else:
            a_logit = None
            pooled = f.mean((2, 3))
        out = self.head(pooled)
        return (out, a_logit) if return_map else out


VARIANT_SPEC = {"baseline": (False, False), "FFN": (True, True),
                "BS-only": (False, True), "AP-only": (True, False)}     # (attention, swap)


def train_model(D, variant, tr, va, s):
    attention, swap = VARIANT_SPEC[variant]
    Y = D[3]
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    model = FruitFocusedNet(attention).to(DEV)
    tr_ld = loader(D, tr, train=True, swap_pool=tr if swap else None)
    va_ld = loader(D, va)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))
    best, best_state = 1e9, None
    for ep in range(CFG["epochs"]):
        model.train()
        for i, (x, y, m7) in enumerate(tr_ld):
            x, y, m7 = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True), m7.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
                out, a_logit = model(x, return_map=True)
                loss = F.cross_entropy(out, y)
                if a_logit is not None:
                    loss = loss + CFG["att_weight"] * F.binary_cross_entropy_with_logits(a_logit.float(), m7)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if (i + 1) % CFG["log_every"] == 0:
                log(f"      epoch {ep+1} batch {i+1}/{len(tr_ld)} loss {loss.item():.4f}")
        sched.step()
        p, yv = predict(model, va_ld, logits=True)
        vloss = F.cross_entropy(p, yv).item()
        log(f"      epoch {ep+1:02d} val NLL {vloss:.4f} acc {(p.argmax(1) == yv).float().mean():.4f}")
        if vloss < best:
            best, best_state = vloss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, ld, logits=False):
    model.eval()
    P, Ys = [], []
    for x, y, _ in ld:
        with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
            P.append(model(x.to(DEV)).float().cpu())
        Ys.append(y)
    P, Ys = torch.cat(P), torch.cat(Ys)
    return (P, Ys) if logits else (P.argmax(1).numpy(), Ys.numpy())


def bal(y, p):
    return float(balanced_accuracy_score(y, p))


def evaluate(model, D, te):
    pn, y = predict(model, loader(D, te))
    ps, _ = predict(model, loader(D, te, pair=opposite_pairs(D[3], te)))
    return dict(natural=bal(y, pn), swapped=bal(y, ps), flip=float((pn != ps).mean()),
                recall_natural=recall_score(y, pn, average=None, labels=[0, 1], zero_division=0).tolist(),
                recall_swapped=recall_score(y, ps, average=None, labels=[0, 1], zero_division=0).tolist())


# --------------------------------------------------------------- datasets
def use_dataset(name):
    global DS, QUALITY_KEYS, FRUITS
    DS = DATASETS[name]
    QUALITY_KEYS, FRUITS = DS["quality_keys"], DS["fruits"]


def prepare(name, masks=True):
    """Load one dataset: photos, labels, fruit names, near-duplicate groups,
    and (if masks) CLIPSeg fruit masks + cleaned backgrounds for swapping."""
    use_dataset(name)
    log(f"=== preparing {name}")
    rows = [r for r in index_images() if r["quality"] in CLASSES]
    assert rows, f"No {name} images under {CFG['input_root']} - did you Add Input?"
    log(f"{name}: images per fruit (Good/Bad): " + ", ".join(
        f"{f} {sum(r['fruit'] == f and r['quality'] == 'Good' for r in rows)}/"
        f"{sum(r['fruit'] == f and r['quality'] == 'Bad' for r in rows)}" for f in FRUITS))
    X = load_cache(rows)
    Y = np.array([CLASSES.index(r["quality"]) for r in rows])
    fruit_arr = np.array([r["fruit"] for r in rows])
    if masks:
        log(f"{name}: segmenting fruit with CLIPSeg")
        M, _ = segment(X, fruit_arr)
        Mg = grow(M, CFG["grow_px"])
        log(f"{name}: cleaning backgrounds")
        Bg = clean_backgrounds(X, Mg)
        del Mg
    else:
        M = np.zeros(X.shape[:3], np.uint8)
        Bg = X                                       # never used (no swapping on this data)
    D = (X, M, Bg, Y)
    log(f"{name}: DINOv2 features for near-duplicate grouping")
    import timm
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                             num_classes=0, img_size=CFG["img"]).eval().to(DEV)
    feats = []
    with torch.no_grad():
        ld = loader(D, np.arange(len(Y)), bs=128)
        for i, (x, _, _) in enumerate(ld):
            with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
                feats.append(dino(x.to(DEV)).float().cpu())
            progress(i, len(ld), "DINOv2 batches", 40)
    del dino
    torch.cuda.empty_cache()
    groups, _ = near_duplicate_groups(torch.cat(feats), CFG["dup_threshold"])
    splits, lofo = make_splits(Y, fruit_arr, groups)
    log(f"{name}: {len(set(groups))} independent near-duplicate groups among {len(Y)} photos")
    return dict(name=name, D=D, Y=Y, fruit=fruit_arr, groups=groups, splits=splits, lofo=lofo,
                fruits=list(FRUITS))


# ---------------------------------------------------------------- summary
def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{np.mean(vals):.3f}" + (f"±{np.std(vals, ddof=1):.3f}" if len(vals) > 1 else "")


def write_summary(res, path):
    L = ["Fruit-Focused Network (FFN) vs baseline. Balanced accuracy, chance 0.500.",
         "Values are mean±sd over the seeds that have finished.", ""]
    for task in ["seeds", "fr_lofo"]:
        if task not in res:
            continue
        name = "FruitNet" if task == "seeds" else "Fresh-Rotten"
        fr = sorted({f for v in res[task].values() for s in v.values() for f in s})
        L.append(f"[{task}] {name} leave-one-fruit-out  (seeds: "
                 + ", ".join(f"{v}={sorted(res[task][v])}" for v in res[task]) + ")")
        L.append(f"  {'variant':10s}{'natural':>14s}{'swapped':>14s}{'flip':>14s}   per fruit (natural)")
        for v, by_seed in res[task].items():
            mean_of = lambda k: [np.mean([r[k] for r in s.values()]) for s in by_seed.values() if s]
            per = "  ".join(f"{f[:6]} {ms([s[f]['natural'] for s in by_seed.values() if f in s])}" for f in fr)
            L.append(f"  {v:10s}{ms(mean_of('natural')):>14s}{ms(mean_of('swapped')):>14s}"
                     f"{ms(mean_of('flip')):>14s}   {per}")
        L.append("")
    if "crossdata" in res:
        L.append("[crossdata] train: ALL FruitNet  ->  test: REAL Fresh-Rotten photos (no pasting)")
        fr = sorted({f for v in res["crossdata"].values() for s in v.values() for f in s["per_fruit"]})
        L.append(f"  {'variant':10s}{'FruitNet test':>16s}{'FN swapped':>14s}{'Fresh-Rotten':>16s}   "
                 + "".join(f"{f:>16s}" for f in fr))
        for v, by_seed in res["crossdata"].items():
            sd = list(by_seed.values())
            L.append(f"  {v:10s}{ms([s['fruitnet_natural'] for s in sd]):>16s}"
                     f"{ms([s['fruitnet_swapped'] for s in sd]):>14s}{ms([s['cross_all'] for s in sd]):>16s}   "
                     + "".join(f"{ms([s['per_fruit'][f] for s in sd]):>16s}" for f in fr))
        L.append("  Recall Good/Bad on Fresh-Rotten (mean over seeds):")
        for v, by_seed in res["crossdata"].items():
            rec = np.mean([s["cross_recall"] for s in by_seed.values()], 0)
            L.append(f"    {v:10s}{rec[0]:.3f}/{rec[1]:.3f}")
    txt = "\n".join(L)
    open(path, "w").write(txt)
    json.dump(res, open(path.replace("summary.txt", "results.json"), "w"), indent=1)
    return txt


# ------------------------------------------------------------------ tasks
def run_lofo(task, P, seeds, res, spath):
    res.setdefault(task, {})
    for v in CFG["variants"]:
        for seed in seeds:
            out = res[task].setdefault(v, {}).setdefault(str(seed), {})
            for fd in P["lofo"]:
                log(f"### {task} | {v} | seed {seed} | test {fd['test']} ({len(fd['tr'])} train)")
                model = train_model(P["D"], v, fd["tr"], fd["va"], seed)
                out[fd["test"]] = r = evaluate(model, P["D"], fd["te"])
                log(f"  -> natural {r['natural']:.3f} swapped {r['swapped']:.3f} flip {r['flip']:.3f}")
                write_summary(res, spath)


def run_cross(FN, FR, seeds, res, spath):
    res.setdefault("crossdata", {})
    sp = FN["splits"]["group"]
    for v in CFG["variants"]:
        for seed in seeds:
            log(f"### crossdata | {v} | seed {seed} | train all FruitNet ({len(sp['tr'])} photos)")
            model = train_model(FN["D"], v, sp["tr"], sp["va"], seed)
            fn = evaluate(model, FN["D"], sp["te"])
            p, y = predict(model, loader(FR["D"], np.arange(len(FR["Y"]))))
            per = {f: bal(y[FR["fruit"] == f], p[FR["fruit"] == f]) for f in FR["fruits"]}
            res["crossdata"].setdefault(v, {})[str(seed)] = dict(
                fruitnet_natural=fn["natural"], fruitnet_swapped=fn["swapped"],
                cross_all=bal(y, p), per_fruit=per,
                cross_recall=recall_score(y, p, average=None, labels=[0, 1], zero_division=0).tolist())
            log(f"  -> FruitNet {fn['natural']:.3f} | Fresh-Rotten {bal(y, p):.3f} "
                + " ".join(f"{f} {a:.3f}" for f, a in per.items()))
            write_summary(res, spath)


def main():
    tasks = CFG["tasks"]
    res, spath = {}, f"{CFG['out']}/summary.txt"
    FR = None
    if "crossdata" in tasks or "fr_lofo" in tasks:
        FR = prepare("freshrotten", masks=("fr_lofo" in tasks))
    if "fr_lofo" in tasks:
        use_dataset("freshrotten")
        run_lofo("fr_lofo", FR, CFG["seeds_fr"], res, spath)
        if "crossdata" not in tasks:
            FR = None
        else:                                        # crossdata only needs the photos: free memory
            X, _, _, Yfr = FR["D"]
            FR["D"] = (X, np.zeros(X.shape[:3], np.uint8), X, Yfr)
    if "crossdata" in tasks or "seeds" in tasks:
        FN = prepare("fruitnet", masks=True)
        if "crossdata" in tasks:
            run_cross(FN, FR, CFG["seeds_cross"], res, spath)
        if "seeds" in tasks:
            use_dataset("fruitnet")
            run_lofo("seeds", FN, CFG["seeds_lofo"], res, spath)
    print("\n" + write_summary(res, spath), flush=True)
    log("done")


if __name__ == "__main__":
    main()
