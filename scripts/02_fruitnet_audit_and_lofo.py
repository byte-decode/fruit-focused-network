"""
PILOT 5: FruitNet (Indian fruits, Good / Bad / Mixed quality).

Why: FruQ turned out to be ~27 independent examples (one time-lapse video
per fruit). FruitNet has ~14,700 separate phone photos of 6 fruits.

This pilot does four things:
  A. Data audit: exact-duplicate removal, a near-duplicate scan with
     DINOv2 similarity (burst photos of the same fruit), and a picture
     grid of each fruit x quality (incl. "Mixed", which we still need to see).
     Near-duplicates are joined into groups, and every split keeps a group
     on one side only, so validation/test numbers are honest.
  B. "Standard" evaluation, as most papers do it: all 6 fruits mixed,
     random split (leaky) and group split (honest).
  C. Leave-one-fruit-out (LOFO): train 4 fruits, calibrate on 1 unseen
     fruit, test on 1 unseen fruit; 6 folds.
  D. Calibration (in-domain vs unseen-fruit) and "send to human" scores
     (MSP vs kNN), same as before.
Task: Good vs Bad (binary). Mixed is shown in the grid but not used yet.
Models: frozen DINOv2-small + linear head (fast), MobileNetV3 fine-tuned.

Kaggle: Add Input -> "shashwatwork/fruitnet-indian-fruits-dataset-with-quality",
Internet ON (for model weights), GPU T4.
Run interactively so you see progress:  !python -u pilot5_fruitnet.py
(progress.txt in the output folder also shows how far it got)
Outputs: /kaggle/working/pilot5/
  summary.txt, results.json, data_audit.txt, grid.jpg, dino_features.npz
"""
print("pilot5: starting, importing libraries ...", flush=True)
import os, json, random, copy, hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (f1_score, roc_auc_score, balanced_accuracy_score,
                             recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from PIL import Image

CFG = dict(
    input_root="/kaggle/input",
    cache_size=256, img=224, bs=64, epochs=12, lr=3e-4, wd=1e-4,
    split_seed=42,
    dup_threshold=0.95,          # DINOv2 cosine above this = near-duplicate
    run_dino=True, dino_C=0.5,
    run_mobilenet=True, mobilenet_seeds=[0],
    max_bank=6000, knn_k=10, n_bins=15,
    save_weights=False,
    workers=2,                   # DataLoader worker processes
    log_every=50,                # print progress every N batches
    out="/kaggle/working/pilot5",
)
os.makedirs(CFG["out"], exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

import time
_T0 = time.time()
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
QUALITY_KEYS = {"Good": ["good"], "Bad": ["bad"], "Mixed": ["mixed"]}
FRUITS = ["apple", "banana", "guava", "lime", "orange", "pomegranate"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------- data
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


def save_grid(X, rows, path, per=2):
    S = 128
    qs = ["Good", "Bad", "Mixed"]
    canvas = Image.new("RGB", (S * per * len(qs), S * len(FRUITS)), "white")
    rng = random.Random(0)
    for i, fr in enumerate(FRUITS):
        for j, q in enumerate(qs):
            idx = [k for k, r in enumerate(rows) if r["fruit"] == fr and r["quality"] == q]
            for k, ix in enumerate(rng.sample(idx, min(per, len(idx)))):
                canvas.paste(Image.fromarray(X[ix]).resize((S, S)), ((j * per + k) * S, i * S))
    canvas.save(path, quality=90)
    print(f"Grid saved: rows = {FRUITS}, columns = {per} x {qs}")


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF_TRAIN = T.Compose([T.RandomResizedCrop(CFG["img"], scale=(0.7, 1.0), antialias=True),
                      T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2), T.Normalize(MEAN, STD)])
TF_EVAL = T.Compose([T.Resize((CFG["img"], CFG["img"]), antialias=True), T.Normalize(MEAN, STD)])


class Cached(Dataset):
    def __init__(self, X, Y, idx, train):
        self.X, self.Y, self.idx, self.train = X, Y, np.asarray(idx), train
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(self.X[j]).permute(2, 0, 1).float().div(255)
        return (TF_TRAIN if self.train else TF_EVAL)(x), int(self.Y[j])


def loader(X, Y, idx, train, bs=None):
    return DataLoader(Cached(X, Y, idx, train), batch_size=bs or CFG["bs"], shuffle=train,
                      num_workers=CFG["workers"], pin_memory=True)


# --------------------------------------------- DINOv2 + duplicate audit
@torch.no_grad()
def dino_features(X):
    try:
        import timm
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0, img_size=CFG["img"])
        print("DINOv2-small loaded from timm")
    except Exception as e:
        print(f"timm DINOv2 failed ({e}) -> torch.hub")
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    m = m.eval().to(DEV)
    feats = []
    ld = loader(X, np.zeros(len(X), int), np.arange(len(X)), False, bs=128)
    for i, (x, _) in enumerate(ld):
        with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
            feats.append(m(x.to(DEV)).float().cpu())
        progress(i, len(ld), "DINOv2 batches", 20)
    del m
    torch.cuda.empty_cache()
    return torch.cat(feats)


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


def audit(rows_all, rows, groups, nn_sim, path):
    qa = np.array([r["quality"] for r in rows_all]); fa = np.array([r["fruit"] for r in rows_all])
    L = ["Images per fruit (Good / Bad / Mixed), after exact-duplicate removal:"]
    for f in FRUITS:
        L.append(f"  {f:12s}" + "".join(f"{np.sum((fa == f) & (qa == k)):7d}" for k in ["Good", "Bad", "Mixed"]))
    L.append("Duplicate scan below covers Good + Bad only (Mixed not used yet).")
    q = np.array([r["quality"] for r in rows]); fr = np.array([r["fruit"] for r in rows])
    sizes = Counter(groups)
    multi = sum(v for v in sizes.values() if v > 1)
    L += ["", f"Near-duplicates (DINOv2 cosine >= {CFG['dup_threshold']}):",
          f"  images with a near-duplicate: {multi} / {len(rows)} ({100 * multi / len(rows):.1f}%)",
          f"  independent groups: {len(sizes)} | largest group: {max(sizes.values())}",
          f"  nearest-neighbour similarity percentiles 50/90/99: "
          f"{np.percentile(nn_sim, 50):.3f} / {np.percentile(nn_sim, 90):.3f} / {np.percentile(nn_sim, 99):.3f}"]
    mixed_groups = sum(1 for g in set(groups)
                       if len({(fr[i], q[i]) for i in np.where(groups == g)[0]}) > 1)
    L.append(f"  groups spanning more than one fruit/quality: {mixed_groups}")
    txt = "\n".join(L)
    open(path, "w").write(txt)
    print(txt)


# --------------------------------------------------------- splits
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
        train_f = [f for f in order if f not in (test_f, calib_f)]
        tr_all = np.where(np.isin(fruit_arr, train_f))[0]
        tr, va = group_split(tr_all, Y, fruit_arr, groups, 0.15, s)
        lofo.append(dict(fold=i, test=test_f, calib=calib_f, train=train_f, tr=tr, va=va,
                         ca=np.where(fruit_arr == calib_f)[0], te=np.where(fruit_arr == test_f)[0]))
    return splits, lofo


# ----------------------------------------------------- evaluation
def fit_temperature(logits, y):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return log_t.exp().item()


def ece_score(probs, y, n_bins=CFG["n_bins"]):
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(correct[m].mean().item() - conf[m].mean().item())
    return ece


def knn_score(h, bank):
    hn = F.normalize(h, dim=1)
    return torch.cat([(c @ bank.T).topk(CFG["knn_k"], dim=1).values[:, -1] for c in hn.split(1024)])


def make_bank(h_tr):
    idx = torch.randperm(len(h_tr), generator=torch.Generator().manual_seed(0))[:CFG["max_bank"]]
    return F.normalize(h_tr[idx], dim=1)


def rejection_metrics(score, correct):
    score, correct = score.numpy(), correct.numpy().astype(float)
    c_sorted = correct[np.argsort(-score)]
    k = int(round(len(c_sorted) * 0.8))
    return dict(auroc=roc_auc_score(correct, score) if 0 < correct.mean() < 1 else float("nan"),
                aurc=float((np.cumsum(1 - c_sorted) / np.arange(1, len(c_sorted) + 1)).mean()),
                acc_rej20=float(c_sorted[:k].mean()))


def basic(lte, yte):
    y, p = yte.numpy(), lte.argmax(1).numpy()
    return dict(acc=float((p == y).mean()), bal_acc=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average="macro", labels=[0, 1]),
                recall=recall_score(y, p, average=None, labels=[0, 1], zero_division=0).tolist(),
                ece=ece_score(F.softmax(lte, 1), yte), n=len(y))


def evaluate_lofo(lva, yva, lca, yca, lte, yte, hte, bank):
    T_in, T_un = fit_temperature(lva, yva), fit_temperature(lca, yca)
    correct = lte.argmax(1) == yte
    r = basic(lte, yte)
    r.update(T_in=T_in, T_unseen=T_un,
             in_domain_val_acc=(lva.argmax(1) == yva).float().mean().item(),
             ece_all={"none": ece_score(F.softmax(lte, 1), yte),
                      "TS in-domain": ece_score(F.softmax(lte / T_in, 1), yte),
                      "TS unseen-fruit": ece_score(F.softmax(lte / T_un, 1), yte)},
             rejection={"MSP": rejection_metrics(F.softmax(lte, 1).max(1).values, correct),
                        "kNN": rejection_metrics(knn_score(hte, bank), correct)})
    return r


# --------------------------------------------------------- models
class DinoProbe:
    def __init__(self, Fe, Y, tr):
        log(f"  fitting linear head on {len(tr)} images")
        self.Fe = Fe
        self.sc = StandardScaler().fit(Fe[tr].numpy())
        self.clf = LogisticRegression(C=CFG["dino_C"], max_iter=5000).fit(
            self.sc.transform(Fe[tr].numpy()), Y[tr])
    def logits(self, ix):
        d = self.clf.decision_function(self.sc.transform(self.Fe[ix].numpy()))
        d = torch.tensor(d, dtype=torch.float32)
        return torch.stack([-d / 2, d / 2], 1)          # binary -> 2-class logits
    def feats(self, ix):
        return self.Fe[ix]


def build_model(nc):
    m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1, quantize=False)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, nc)
    return m


def forward_feats(model, x):
    h = model.classifier[:3](model.avgpool(model.features(x)).flatten(1))
    return h, model.classifier[3](h)


@torch.no_grad()
def run_model(model, ld):
    model.eval().to(DEV)
    H, L = [], []
    for x, _ in ld:
        h, lg = forward_feats(model, x.to(DEV))
        H.append(h.float().cpu()); L.append(lg.float().cpu())
    return torch.cat(H), torch.cat(L)


def train_one(X, Y, tr, va, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_model(2).to(DEV)
    tr_ld, va_ld = loader(X, Y, tr, True), loader(X, Y, va, False)
    yv = torch.from_numpy(Y[va])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))
    best, best_state = 1e9, None
    for ep in range(CFG["epochs"]):
        model.train()
        for i, (x, y) in enumerate(tr_ld):
            if ep == 0 and i == 0:
                log(f"    first training batch ready ({len(tr_ld)} batches per epoch)")
            if (i + 1) % CFG["log_every"] == 0:
                log(f"    epoch {ep+1} batch {i+1}/{len(tr_ld)}")
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        _, lg = run_model(model, va_ld)
        vloss = F.cross_entropy(lg, yv).item()
        log(f"    epoch {ep+1:02d}  val NLL {vloss:.4f}  acc {(lg.argmax(1) == yv).float().mean():.4f}")
        if vloss < best:
            best, best_state = vloss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


class MobileNetRunner:
    def __init__(self, X, Y, tr, va, seed):
        log(f"  training MobileNetV3: {len(tr)} train / {len(va)} val images")
        self.X, self.Y = X, Y
        self.model = train_one(X, Y, tr, va, seed)
        self._cache = {}
    def _run(self, ix):
        key = (len(ix), int(ix[0]), int(ix[-1]))
        if key not in self._cache:
            self._cache[key] = run_model(self.model, loader(self.X, self.Y, ix, False))
        return self._cache[key]
    def logits(self, ix):
        return self._run(ix)[1]
    def feats(self, ix):
        return self._run(ix)[0]


# --------------------------------------------------------- experiments
def run_all(make_model, name, Y, splits, lofo, results, seed=0):
    Yt = torch.from_numpy(Y)
    for sname in ["random", "group"]:
        sp = splits[sname]
        m = make_model(sp["tr"], sp["va"], seed)
        r = basic(m.logits(sp["te"]), Yt[sp["te"]])
        r.update(model=name, seed=seed, setting=f"standard-{sname}")
        results.append(r)
        log(f"{name} standard-{sname} done")
        print(f"  {name} standard-{sname}: acc {r['acc']:.4f} bal {r['bal_acc']:.4f}")
    for fd in lofo:
        m = make_model(fd["tr"], fd["va"], seed)
        bank_ix = np.random.RandomState(0).permutation(fd["tr"])[:CFG["max_bank"]]
        r = evaluate_lofo(m.logits(fd["va"]), Yt[fd["va"]], m.logits(fd["ca"]), Yt[fd["ca"]],
                          m.logits(fd["te"]), Yt[fd["te"]], m.feats(fd["te"]),
                          F.normalize(m.feats(np.sort(bank_ix)), dim=1))
        r.update(model=name, seed=seed, setting="LOFO", fold=fd["fold"], test=fd["test"], calib=fd["calib"])
        results.append(r)
        log(f"{name} LOFO fold {fd['fold']} done")
        print(f"  {name} LOFO test {fd['test']:12s}: acc {r['acc']:.4f} bal {r['bal_acc']:.4f} "
              f"recall G/B {np.round(r['recall'], 2)}")


def write_summary(results, path):
    L = ["FruitNet, Good vs Bad. bal_acc = mean recall of Good and Bad (chance 0.5).", ""]
    models = list(dict.fromkeys(r["model"] for r in results))
    L.append("STANDARD evaluation (all 6 fruits seen in training):")
    L.append(f"  {'model':22s}{'split':>8s}{'acc':>8s}{'bal_acc':>9s}{'rec G/B':>12s}{'n test':>8s}")
    for m in models:
        for r in [r for r in results if r["model"] == m and r["setting"].startswith("standard")]:
            rec = "/".join(f"{v:.2f}" for v in r["recall"])
            L.append(f"  {m:22s}{r['setting'][9:]:>8s}{r['acc']:>8.4f}{r['bal_acc']:>9.4f}{rec:>12s}{r['n']:>8d}")
    L += ["", "LEAVE-ONE-FRUIT-OUT (test fruit never seen in training):"]
    for m in models:
        rs = [r for r in results if r["model"] == m and r["setting"] == "LOFO"]
        if not rs:
            continue
        L.append(f"\n  {m}")
        L.append(f"  {'test':12s}{'calib':>12s}{'acc':>7s}{'bal':>7s}{'rec G/B':>11s}{'val acc':>9s}"
                 f"{'ECE none/in/unseen':>22s}{'T in/unseen':>16s}{'AUROC MSP/kNN':>16s}{'rej20 MSP/kNN':>15s}")
        for r in rs:
            e, rj = r["ece_all"], r["rejection"]
            rec = "/".join(f"{v:.2f}" for v in r["recall"])
            L.append(f"  {r['test']:12s}{r['calib']:>12s}{r['acc']:>7.3f}{r['bal_acc']:>7.3f}{rec:>11s}"
                     f"{r['in_domain_val_acc']:>9.3f}"
                     f"{e['none']:>10.3f}/{e['TS in-domain']:.3f}/{e['TS unseen-fruit']:.3f}"
                     f"{min(r['T_in'], 9999):>9.2f}/{min(r['T_unseen'], 9999):<7.2f}"
                     f"{rj['MSP']['auroc']:>10.3f}/{rj['kNN']['auroc']:.3f}"
                     f"{rj['MSP']['acc_rej20']:>9.3f}/{rj['kNN']['acc_rej20']:.3f}")
        mean = lambda f: np.nanmean([f(r) for r in rs])
        L.append(f"  {'MEAN':12s}{'':>12s}{mean(lambda r: r['acc']):>7.3f}{mean(lambda r: r['bal_acc']):>7.3f}"
                 f"{'':>11s}{mean(lambda r: r['in_domain_val_acc']):>9.3f}"
                 f"{mean(lambda r: r['ece_all']['none']):>10.3f}/{mean(lambda r: r['ece_all']['TS in-domain']):.3f}"
                 f"/{mean(lambda r: r['ece_all']['TS unseen-fruit']):.3f}{'':>16s}"
                 f"{mean(lambda r: r['rejection']['MSP']['auroc']):>10.3f}/{mean(lambda r: r['rejection']['kNN']['auroc']):.3f}"
                 f"{mean(lambda r: r['rejection']['MSP']['acc_rej20']):>9.3f}/{mean(lambda r: r['rejection']['kNN']['acc_rej20']):.3f}")
        wins = sum(r["ece_all"]["TS unseen-fruit"] < r["ece_all"]["TS in-domain"] for r in rs)
        knn = sum(r["rejection"]["kNN"]["aurc"] < r["rejection"]["MSP"]["aurc"] for r in rs)
        L.append(f"  unseen-fruit TS beats in-domain TS: {wins}/{len(rs)} | kNN beats MSP (AURC): {knn}/{len(rs)}")
    txt = "\n".join(L)
    open(path, "w").write(txt)
    json.dump(results, open(path.replace("summary.txt", "results.json"), "w"), indent=1)
    return txt


# ---------------------------------------------------------------- main
def main():
    log("indexing images")
    rows_all = index_images()
    assert rows_all, f"No FruitNet images found under {CFG['input_root']} - did you Add Input?"
    log(f"{len(rows_all)} images indexed; loading into memory")
    X_all = load_cache(rows_all)
    save_grid(X_all, rows_all, f"{CFG['out']}/grid.jpg")

    keep = np.array([r["quality"] in CLASSES for r in rows_all])
    rows = [r for r, k in zip(rows_all, keep) if k]
    X = X_all[keep]; del X_all
    Y = np.array([CLASSES.index(r["quality"]) for r in rows])
    fruit_arr = np.array([r["fruit"] for r in rows])

    fpath = f"{CFG['out']}/dino_features.npz"
    if os.path.exists(fpath):
        Fe = torch.from_numpy(np.load(fpath)["features"])
    else:
        log("extracting DINOv2 features")
        Fe = dino_features(X)
    log("near-duplicate scan")
    groups, nn_sim = near_duplicate_groups(Fe, CFG["dup_threshold"])
    np.savez_compressed(fpath, features=Fe.numpy(), labels=Y, fruits=fruit_arr, groups=groups,
                        nn_sim=nn_sim, paths=np.array([os.path.relpath(r["path"], CFG["input_root"]) for r in rows]))
    print()
    audit(rows_all, rows, groups, nn_sim, f"{CFG['out']}/data_audit.txt")

    log("making splits")
    splits, lofo = make_splits(Y, fruit_arr, groups)
    for fd in lofo:
        print(f"LOFO fold {fd['fold']}: test {fd['test']:12s} calib {fd['calib']:12s} "
              f"n train/val/calib/test {len(fd['tr'])}/{len(fd['va'])}/{len(fd['ca'])}/{len(fd['te'])}")

    results, spath = [], f"{CFG['out']}/summary.txt"
    if CFG["run_dino"]:
        print("\n##### DINOv2 linear probe")
        run_all(lambda tr, va, seed: DinoProbe(Fe, Y, tr), "DINOv2-LP", Y, splits, lofo, results)
        print("\n" + write_summary(results, spath))
    if CFG["run_mobilenet"]:
        for seed in CFG["mobilenet_seeds"]:
            print(f"\n##### MobileNetV3 fine-tune, seed {seed}")
            run_all(lambda tr, va, s: MobileNetRunner(X, Y, tr, va, s),
                    f"MobileNetV3-FT s{seed}", Y, splits, lofo, results, seed)
            print("\n" + write_summary(results, spath))


if __name__ == "__main__":
    main()
