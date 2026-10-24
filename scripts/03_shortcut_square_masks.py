"""
PILOT 6: Shortcut test on FruitNet (Good vs Bad).

Question: does the model judge the FRUIT, or the way the photo was taken
(hand, beige wall, background)? The same frozen DINOv2 + linear head is
trained on differently masked versions of every photo:

  full            the whole photo (pilot 5 setting)
  background-60   centre 60% of the photo greyed out (fruit mostly hidden)
  background-75   centre 75% greyed out (only a thin outer border remains)
  centre-60       only the centre 60% kept and enlarged (mostly fruit)
  border-colour   no deep model: 24-number colour histogram of the outer
                  15% ring -> logistic regression

If the background-only versions still separate Good from Bad far above
chance (0.5), the dataset contains a capture shortcut.

Evaluated with the honest near-duplicate-aware split (all fruits seen)
and leave-one-fruit-out. Runs in ~10-15 min on a T4.

Kaggle: Add Input -> shashwatwork/fruitnet-indian-fruits-dataset-with-quality,
Internet ON, GPU T4.   Run:  !python -u pilot6_shortcut.py
Outputs: /kaggle/working/pilot6/ (summary.txt, results.json, masks_example.jpg,
features_<variant>.npz, progress.txt)
"""
print("pilot6: starting, importing libraries ...", flush=True)
import os, json, random, hashlib, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
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
    ring=0.15,                   # outer ring width for the border-colour baseline
    workers=2,
    out="/kaggle/working/pilot6",
)
os.makedirs(CFG["out"], exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_T0 = time.time()

CLASSES = ["Good", "Bad"]
QUALITY_KEYS = {"Good": ["good"], "Bad": ["bad"], "Mixed": ["mixed"]}
FRUITS = ["apple", "banana", "guava", "lime", "orange", "pomegranate"]
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




# ------------------------------------------------------------ variants
VARIANTS = ["full", "background-60", "background-75", "centre-60"]


def make_view(x, variant):
    """x: float tensor 3xSxS in [0,1] -> normalised 3x224x224 view."""
    S = x.shape[-1]
    if variant.startswith("background"):
        frac = int(variant.split("-")[1]) / 100
        a = int(S * (1 - frac) / 2)
        x = x.clone()
        x[:, a:S - a, a:S - a] = torch.tensor(MEAN).view(3, 1, 1)   # mean grey
    elif variant == "centre-60":
        a = int(S * 0.2)
        x = x[:, a:S - a, a:S - a]
    x = TF.resize(x, [CFG["img"], CFG["img"]], antialias=True)
    return TF.normalize(x, MEAN, STD)


class Views(Dataset):
    def __init__(self, X, variant):
        self.X, self.variant = X, variant
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).permute(2, 0, 1).float().div(255)
        return make_view(x, self.variant)


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
def features(model, X, variant):
    path = f"{CFG['out']}/features_{variant}.npz"
    if os.path.exists(path):
        return torch.from_numpy(np.load(path)["features"])
    ld = DataLoader(Views(X, variant), batch_size=128, num_workers=CFG["workers"], pin_memory=True)
    out = []
    for i, x in enumerate(ld):
        with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
            out.append(model(x.to(DEV)).float().cpu())
        progress(i, len(ld), f"DINOv2 {variant}", 30)
    Fe = torch.cat(out)
    np.savez_compressed(path, features=Fe.numpy())
    return Fe


def border_colour_features(X):
    """24 numbers per image: 8-bin histogram of R, G, B over the outer ring."""
    S = X.shape[1]
    r = int(S * CFG["ring"])
    mask = np.ones((S, S), bool)
    mask[r:S - r, r:S - r] = False
    out = np.empty((len(X), 24), np.float32)
    for i in range(len(X)):
        px = X[i][mask]                                   # (n_pixels, 3)
        out[i] = np.concatenate([np.histogram(px[:, c], bins=8, range=(0, 256))[0]
                                 for c in range(3)]) / len(px)
        progress(i, len(X), "border colour", 5000)
    return torch.from_numpy(out)


def save_masks_example(X, rows, path):
    rng = random.Random(1)
    picks = []
    for q in CLASSES:
        idx = [i for i, r in enumerate(rows) if r["quality"] == q]
        picks += rng.sample(idx, 3)
    S = 160
    canvas = Image.new("RGB", (S * len(VARIANTS), S * len(picks)), "white")
    for r_i, ix in enumerate(picks):
        x = torch.from_numpy(X[ix]).permute(2, 0, 1).float().div(255)
        for c_i, v in enumerate(VARIANTS):
            y = make_view(x, v) * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
            im = Image.fromarray((y.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            canvas.paste(im.resize((S, S)), (c_i * S, r_i * S))
    canvas.save(path, quality=90)
    log(f"mask example saved: columns {VARIANTS}, rows 3 Good then 3 Bad")


# ----------------------------------------------------------- evaluation
def probe(Fe, Y, tr, te):
    sc = StandardScaler().fit(Fe[tr].numpy())
    clf = LogisticRegression(C=CFG["dino_C"], max_iter=5000).fit(sc.transform(Fe[tr].numpy()), Y[tr])
    p = clf.predict(sc.transform(Fe[te].numpy()))
    return dict(acc=float((p == Y[te]).mean()), bal_acc=balanced_accuracy_score(Y[te], p),
                recall=recall_score(Y[te], p, average=None, labels=[0, 1], zero_division=0).tolist())


def run_variant(name, Fe, Y, splits, lofo):
    res = dict(variant=name)
    for s in ["random", "group"]:
        sp = splits[s]
        res[f"standard-{s}"] = probe(Fe, Y, np.concatenate([sp["tr"], sp["va"]]), sp["te"])
    res["LOFO"] = {fd["test"]: probe(Fe, Y, np.concatenate([fd["tr"], fd["va"]]), fd["te"]) for fd in lofo}
    log(f"{name:15s} standard-group bal acc {res['standard-group']['bal_acc']:.3f} | LOFO mean "
        f"{np.mean([r['bal_acc'] for r in res['LOFO'].values()]):.3f}")
    return res


def write_summary(results, path, n_img):
    fr = list(results[0]["LOFO"].keys())
    L = [f"FruitNet Good vs Bad, {n_img} images. Balanced accuracy (chance = 0.500).",
         "Classifier: frozen DINOv2-small + logistic regression (border-colour: colour histogram only).",
         "", f"{'input the model sees':18s}{'random':>8s}{'group':>8s}{'LOFO':>8s}   " + "".join(f"{f[:7]:>8s}" for f in fr)]
    for r in results:
        lo = [r["LOFO"][f]["bal_acc"] for f in fr]
        L.append(f"{r['variant']:18s}{r['standard-random']['bal_acc']:>8.3f}{r['standard-group']['bal_acc']:>8.3f}"
                 f"{np.mean(lo):>8.3f}   " + "".join(f"{v:>8.3f}" for v in lo))
    L += ["", "random / group = standard evaluation, all 6 fruits in training (group = near-duplicates kept",
          "on one side). LOFO = mean over leave-one-fruit-out folds; per-fruit columns = the test fruit.",
          "", "Recall Good/Bad, standard-group split:"]
    for r in results:
        L.append(f"  {r['variant']:18s}" + "/".join(f"{v:.3f}" for v in r["standard-group"]["recall"]))
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
    save_masks_example(X, rows, f"{CFG['out']}/masks_example.jpg")

    model = load_dino()
    feats = {}
    for v in VARIANTS:
        log(f"extracting DINOv2 features: {v}")
        feats[v] = features(model, X, v)
    del model
    torch.cuda.empty_cache()

    log("near-duplicate scan (on full photos) for honest splits")
    groups, _ = near_duplicate_groups(feats["full"], CFG["dup_threshold"])
    splits, lofo = make_splits(Y, fruit_arr, groups)

    log("border-colour features")
    feats["border-colour"] = border_colour_features(X)

    results, spath = [], f"{CFG['out']}/summary.txt"
    for v in VARIANTS + ["border-colour"]:
        results.append(run_variant(v, feats[v], Y, splits, lofo))
        write_summary(results, spath, len(Y))
    print("\n" + write_summary(results, spath, len(Y)), flush=True)
    log("done")


if __name__ == "__main__":
    main()
