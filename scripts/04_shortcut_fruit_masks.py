"""
PILOT 7: Fruit masks -> airtight shortcut test + a fix (FruitNet, Good vs Bad).

Pilot 6 showed the photo border alone predicts Good/Bad (85%), but a grey
square does not always hide the fruit. Here every fruit is segmented with
CLIPSeg (text prompt = the fruit name, so the hand is not included), then:

  full             whole photo
  fruit-only       fruit kept, everything else plain grey (no shortcut possible)
  background-only  fruit (mask enlarged) greyed out, rest kept
  mask-geometry    5 numbers only: fruit area, centre x/y, box width/height
                   (does fruit SIZE/POSITION in the frame give the class away?)

Part A (fast, ~20-30 min): frozen DINOv2 + linear head on each version,
  honest group split + leave-one-fruit-out (LOFO).
Part B (~1 h): MobileNetV3 fine-tuned on full vs fruit-only photos, LOFO.
  This is the "fix": does removing the background help on unseen fruits?

Kaggle: Add Input -> shashwatwork/fruitnet-indian-fruits-dataset-with-quality,
Internet ON, GPU T4.   Run:  !python -u pilot7_masks.py
Outputs: /kaggle/working/pilot7/  summary.txt, results.json, mask_check.jpg,
  mask_stats.txt, masks_128.npz, progress.txt
"""
print("pilot7: starting, importing libraries ...", flush=True)
import os, json, random, hashlib, time, copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from PIL import Image

CFG = dict(
    input_root="/kaggle/input",
    cache_size=256, img=224,
    split_seed=42,
    dup_threshold=0.95,
    dino_C=0.5,
    clipseg="CIDAS/clipseg-rd64-refined",
    mask_threshold=0.4,          # CLIPSeg probability above this = fruit
    grow_px=12,                  # enlarge fruit mask by this many pixels for background-only
    run_mobilenet=True, mobilenet_seed=0,
    bs=64, epochs=12, lr=3e-4, wd=1e-4, log_every=50,
    workers=2,
    out="/kaggle/working/pilot7",
)
os.makedirs(CFG["out"], exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_T0 = time.time()

CLASSES = ["Good", "Bad"]
QUALITY_KEYS = {"Good": ["good"], "Bad": ["bad"], "Mixed": ["mixed"]}
FRUITS = ["apple", "banana", "guava", "lime", "orange", "pomegranate"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
GREY = torch.tensor(MEAN).view(3, 1, 1)

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
        train_f = [f for f in order if f not in (test_f, calib_f)]
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
        texts = [f"a {f}" for f in fruit_arr[sl]]
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


def geometry(M):
    """5 numbers per image: area fraction, centre x, centre y, box width, box height."""
    S = M.shape[1]
    g = np.zeros((len(M), 5), np.float32)
    for i, m in enumerate(M):
        ys, xs = np.nonzero(m)
        if len(xs):
            g[i] = [len(xs) / S / S, xs.mean() / S, ys.mean() / S,
                    (xs.max() - xs.min()) / S, (ys.max() - ys.min()) / S]
    return torch.from_numpy(g)


def mask_report(M, Mg, X, Y, fruit_arr, probs_max, path_txt, path_img):
    area = M.reshape(len(M), -1).mean(1)
    L = ["Fruit area (share of photo) from CLIPSeg masks, mean per fruit:",
         f"  {'fruit':12s}{'Good':>8s}{'Bad':>8s}{'low-confidence masks':>24s}"]
    for f in FRUITS:
        g, b = (fruit_arr == f) & (Y == 0), (fruit_arr == f) & (Y == 1)
        low = ((fruit_arr == f) & (probs_max < CFG["mask_threshold"])).sum()
        L.append(f"  {f:12s}{area[g].mean():>8.3f}{area[b].mean():>8.3f}{low:>24d}")
    L.append("(low-confidence = CLIPSeg never reached the threshold; fallback mask used)")
    open(path_txt, "w").write("\n".join(L))
    print("\n".join(L), flush=True)
    # visual check: per fruit x class, 2 photos as  [photo | fruit-only | background-only]
    S = 128
    rng = random.Random(0)
    tiles = []
    for f in FRUITS:
        for c in (0, 1):
            idx = np.where((fruit_arr == f) & (Y == c))[0]
            tiles += list(rng.sample(list(idx), min(2, len(idx))))
    cols = 4
    canvas = Image.new("RGB", (S * 3 * cols, S * ((len(tiles) + cols - 1) // cols)), "white")
    for k, i in enumerate(tiles):
        x = torch.from_numpy(X[i]).permute(2, 0, 1).float().div(255)
        views = [x, apply_mode(x, M[i], Mg[i], "fruit-only"), apply_mode(x, M[i], Mg[i], "background-only")]
        for v_i, v in enumerate(views):
            im = Image.fromarray((v.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((S, S))
            canvas.paste(im, (((k % cols) * 3 + v_i) * S, (k // cols) * S))
    canvas.save(path_img, quality=90)
    log(f"mask check saved: rows = fruits, Good pair then Bad pair; each = photo | fruit-only | background-only")


# --------------------------------------------------------------- views
def apply_mode(x, m, mg, mode):
    """x: 3xSxS float [0,1]; m: fruit mask; mg: grown fruit mask (uint8 arrays)."""
    if mode == "fruit-only":
        k = torch.from_numpy(m).float()[None]
        return x * k + GREY * (1 - k)
    if mode == "background-only":
        k = torch.from_numpy(mg).float()[None]
        return x * (1 - k) + GREY * k
    return x


AUG = T.Compose([T.RandomResizedCrop(CFG["img"], scale=(0.7, 1.0), antialias=True),
                 T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2)])


class MaskedViews(Dataset):
    def __init__(self, X, M, Mg, Y, idx, mode, train=False):
        self.X, self.M, self.Mg, self.Y = X, M, Mg, Y
        self.idx, self.mode, self.train = np.asarray(idx), mode, train
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(self.X[j]).permute(2, 0, 1).float().div(255)
        x = apply_mode(x, self.M[j], self.Mg[j], self.mode)
        x = AUG(x) if self.train else TF.resize(x, [CFG["img"], CFG["img"]], antialias=True)
        return TF.normalize(x, MEAN, STD), int(self.Y[j])


def make_loader(D, idx, mode, train=False, bs=None):
    X, M, Mg, Y = D
    return DataLoader(MaskedViews(X, M, Mg, Y, idx, mode, train), batch_size=bs or CFG["bs"],
                      shuffle=train, num_workers=CFG["workers"], pin_memory=True)


# ------------------------------------------------------------- DINOv2
def load_dino():
    try:
        import timm
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0, img_size=CFG["img"])
        log("DINOv2-small loaded from timm")
    except Exception as e:
        log(f"timm DINOv2 failed ({e}) -> torch.hub")
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    return m.eval().to(DEV)


@torch.no_grad()
def dino_feats(model, D, mode):
    ld = make_loader(D, np.arange(len(D[0])), mode, bs=128)
    out = []
    for i, (x, _) in enumerate(ld):
        with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
            out.append(model(x.to(DEV)).float().cpu())
        progress(i, len(ld), f"DINOv2 {mode}", 30)
    return torch.cat(out)


def probe(Fe, Y, tr, te):
    sc = StandardScaler().fit(Fe[tr].numpy())
    clf = LogisticRegression(C=CFG["dino_C"], max_iter=5000).fit(sc.transform(Fe[tr].numpy()), Y[tr])
    p = clf.predict(sc.transform(Fe[te].numpy()))
    return score(Y[te], p)


def score(y, p):
    return dict(bal_acc=balanced_accuracy_score(y, p), acc=float((p == y).mean()),
                recall=recall_score(y, p, average=None, labels=[0, 1], zero_division=0).tolist())


def run_probe(name, Fe, Y, splits, lofo):
    sp = splits["group"]
    res = dict(model="DINOv2-LP" if name != "mask-geometry" else "LogReg (5 numbers)", input=name,
               group=probe(Fe, Y, np.concatenate([sp["tr"], sp["va"]]), sp["te"]),
               LOFO={fd["test"]: probe(Fe, Y, np.concatenate([fd["tr"], fd["va"]]), fd["te"]) for fd in lofo})
    log(f"{name:16s} group {res['group']['bal_acc']:.3f} | LOFO mean "
        f"{np.mean([r['bal_acc'] for r in res['LOFO'].values()]):.3f}")
    return res


# ---------------------------------------------------------- MobileNetV3
def build_model():
    m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1, quantize=False)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    return m


@torch.no_grad()
def predict(model, ld):
    model.eval()
    return torch.cat([model(x.to(DEV)).float().argmax(1).cpu() for x, _ in ld]).numpy()


def train_eval(D, mode, tr, va, te, seed):
    Y = D[3]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_model().to(DEV)
    tr_ld, va_ld = make_loader(D, tr, mode, train=True), make_loader(D, va, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))
    yv = torch.from_numpy(Y[va])
    best, best_state = 1e9, None
    for ep in range(CFG["epochs"]):
        model.train()
        for i, (x, y) in enumerate(tr_ld):
            if (i + 1) % CFG["log_every"] == 0:
                log(f"      epoch {ep+1} batch {i+1}/{len(tr_ld)}")
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = torch.cat([model(x.to(DEV)).float().cpu() for x, _ in va_ld])
        vloss = F.cross_entropy(lg, yv).item()
        log(f"      epoch {ep+1:02d} val NLL {vloss:.4f} acc {(lg.argmax(1) == yv).float().mean():.4f}")
        if vloss < best:
            best, best_state = vloss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return score(Y[te], predict(model, make_loader(D, te, mode)))


# ------------------------------------------------------------- summary
def write_summary(results, path, n):
    fr = list(results[0]["LOFO"].keys())
    L = [f"FruitNet Good vs Bad, {n} images. Balanced accuracy (chance = 0.500).", "",
         f"{'model':20s}{'input':17s}{'group':>7s}{'LOFO':>7s}   " + "".join(f"{f[:7]:>8s}" for f in fr)]
    for r in results:
        lo = [r["LOFO"][f]["bal_acc"] for f in fr if f in r["LOFO"]]
        L.append(f"{r['model']:20s}{r['input']:17s}"
                 + (f"{r['group']['bal_acc']:>7.3f}" if r.get("group") else f"{'-':>7s}")
                 + f"{np.mean(lo):>7.3f}   " + "".join(
                     f"{r['LOFO'][f]['bal_acc']:>8.3f}" if f in r["LOFO"] else f"{'-':>8s}" for f in fr))
    L += ["", "group = all 6 fruits in training, near-duplicates kept on one side (honest split).",
          "LOFO = leave-one-fruit-out mean; per-fruit columns = the unseen test fruit.",
          "", "Recall Good/Bad per unseen fruit:"]
    for r in results:
        L.append(f"  {r['model'][:18]:18s} {r['input']:16s}" + "  ".join(
            f"{f[:5]} " + "/".join(f"{v:.2f}" for v in r["LOFO"][f]["recall"]) for f in fr if f in r["LOFO"]))
    txt = "\n".join(L)
    open(path, "w").write(txt)
    json.dump(results, open(path.replace("summary.txt", "results.json"), "w"), indent=1)
    return txt


# ---------------------------------------------------------------- main
def main():
    log("indexing images")
    rows = [r for r in index_images() if r["quality"] in CLASSES]
    assert rows, f"No FruitNet images found under {CFG['input_root']} - did you Add Input?"
    log(f"{len(rows)} Good/Bad images; loading into memory")
    X = load_cache(rows)
    Y = np.array([CLASSES.index(r["quality"]) for r in rows])
    fruit_arr = np.array([r["fruit"] for r in rows])

    log("segmenting fruit with CLIPSeg")
    M, pmax = segment(X, fruit_arr)
    Mg = grow(M, CFG["grow_px"])
    m128 = F.max_pool2d(torch.from_numpy(M).float()[:, None], 2)[:, 0].numpy() > 0
    np.savez_compressed(f"{CFG['out']}/masks_128.npz", masks=np.packbits(m128, axis=-1),
                        prob_max=pmax, labels=Y, fruits=fruit_arr)
    mask_report(M, Mg, X, Y, fruit_arr, pmax, f"{CFG['out']}/mask_stats.txt", f"{CFG['out']}/mask_check.jpg")
    D = (X, M, Mg, Y)

    model = load_dino()
    feats = {}
    for mode in ["full", "fruit-only", "background-only"]:
        log(f"extracting DINOv2 features: {mode}")
        feats[mode] = dino_feats(model, D, mode)
    del model
    torch.cuda.empty_cache()
    feats["mask-geometry"] = geometry(M)

    log("near-duplicate scan (full photos) for honest splits")
    groups, _ = near_duplicate_groups(feats["full"], CFG["dup_threshold"])
    splits, lofo = make_splits(Y, fruit_arr, groups)

    results, spath = [], f"{CFG['out']}/summary.txt"
    for mode in ["full", "fruit-only", "background-only", "mask-geometry"]:
        results.append(run_probe(mode, feats[mode], Y, splits, lofo))
        write_summary(results, spath, len(Y))
    print("\n" + write_summary(results, spath, len(Y)), flush=True)
    log("Part A done (DINOv2). Starting Part B (MobileNetV3)." if CFG["run_mobilenet"] else "done")

    if CFG["run_mobilenet"]:
        s = CFG["mobilenet_seed"]
        for mode in ["full", "fruit-only"]:
            r = dict(model="MobileNetV3-FT", input=mode, group=None, LOFO={})
            results.append(r)
            for fd in lofo:
                log(f"  MobileNetV3 {mode}: LOFO test {fd['test']} ({len(fd['tr'])} train images)")
                r["LOFO"][fd["test"]] = train_eval(D, mode, fd["tr"], fd["va"], fd["te"], s)
                log(f"  -> {fd['test']} bal acc {r['LOFO'][fd['test']]['bal_acc']:.3f}")
                write_summary(results, spath, len(Y))
        print("\n" + write_summary(results, spath, len(Y)), flush=True)
        log("done")


if __name__ == "__main__":
    main()
