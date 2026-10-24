"""
PILOT 4: Leave-one-fruit-out (LOFO) on FruQ.

Pilot 3 found that with only 4 training fruits, the model calls almost
every unseen fruit "Rotten" (recall Fresh 33%, Mild 14%, Rotten 96%).
This pilot tests two possible fixes:
  1. More training fruits: train on 7, calibrate on 1, test on 1,
     rotating so every fruit is the test fruit once (9 folds).
  2. A different model: frozen DINOv2-small features + a small linear head
     (with and without class-balanced weighting), compared against
     MobileNetV3 fine-tuned exactly as in pilot 3.

Main numbers: balanced accuracy and per-class recall (plain accuracy is
misleading when the model leans towards one class).

Order of work: DINOv2 runs first (a few minutes) and summary.txt is
written straight away; MobileNetV3 runs after and updates the summary
after every fold, so a timeout still leaves usable results.

Kaggle: Internet ON, GPU T4. Run:  !python pilot4_fruq_lofo.py
Outputs: /kaggle/working/pilot4/
  summary.txt, results.json, dino_features.npz (reusable without a GPU),
  per-run test outputs (.npz), MobileNet weights (.pt)
"""
import os, json, random, copy, shutil, zipfile, hashlib, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, roc_auc_score, balanced_accuracy_score,
                             recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from PIL import Image

CFG = dict(
    zenodo="https://zenodo.org/records/7224690/files/{}?download=1",
    db_zip="FruQ-DB.zip", db_md5="1a942c2d49dc302bacef155561e1f9a8",
    multi_zip="FruQ-multi.zip",
    force_multi=False,
    data_dir="/kaggle/tmp/fruq",
    cache_size=256,            # images are resized once and kept in memory
    img=224, bs=64, epochs=12, lr=3e-4, wd=1e-4,
    split_seed=42,
    run_dino=True, dino_C=0.5,
    run_mobilenet=True, mobilenet_seeds=[0],   # add 1, 2 later if time allows
    max_bank=6000, knn_k=10, n_bins=15,
    save_weights=True,
    out="/kaggle/working/pilot4",
)
os.makedirs(CFG["out"], exist_ok=True)
os.makedirs(CFG["data_dir"], exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["Fresh", "Mild", "Rotten"]
CLASS_KEYS = {"Rotten": ["rotten", "bad"], "Mild": ["mild", "mixed", "medium"],
              "Fresh": ["fresh", "good"]}
FRUITS = ["banana", "cucumber", "grape", "kaki", "persimmon", "papaya", "peach", "pear",
          "pepper", "strawberry", "tomato", "watermelon", "avocado", "apple", "orange",
          "lime", "lemon", "pomegranate", "mango", "carrot", "potato"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PILOT3_REF = "pilot 3 reference (4 training fruits, MobileNetV3): balanced acc 0.480 | recall F/M/R 0.333/0.144/0.963"


# ------------------------------------------------- data (same as pilot 3)
def download(name, md5=None):
    dst = os.path.join(CFG["data_dir"], name)
    if not (os.path.exists(dst) and os.path.getsize(dst) > 1_000_000):
        print(f"Downloading {name} ...")
        with urllib.request.urlopen(CFG["zenodo"].format(name)) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
    if md5:
        h = hashlib.md5()
        with open(dst, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        assert h.hexdigest() == md5, f"MD5 mismatch for {name} - delete it and rerun"
    out = dst[:-4]
    if not os.path.isdir(out):
        print(f"Unzipping {name} ...")
        with zipfile.ZipFile(dst) as z:
            z.extractall(out)
    return out


def find_one(text, table):
    hits = {k for k, words in table.items() if any(w in text for w in words)}
    return hits.pop() if len(hits) == 1 else None


def index_images(root):
    rows, unknown = [], 0
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if not fn.lower().endswith(IMG_EXT) or fn.startswith("._"):
                continue
            path = os.path.join(dp, fn)
            rel = os.path.relpath(path, root).lower()
            cls = find_one(os.path.dirname(rel), CLASS_KEYS) or find_one(rel, CLASS_KEYS)
            hits = [f for f in FRUITS if f in rel]
            fruit = "kaki" if hits == ["persimmon"] else (max(hits, key=len) if hits else None)
            if cls is None:
                unknown += 1
                continue
            rows.append(dict(path=path, label=CLASSES.index(cls), fruit=fruit))
    return rows, unknown


def load_index():
    rows = []
    if not CFG["force_multi"]:
        rows, unk = index_images(download(CFG["db_zip"], CFG["db_md5"]))
        known = sum(r["fruit"] is not None for r in rows)
        print(f"FruQ-DB: {len(rows)} labelled images ({unk} without class), {known} with a fruit name")
        if rows and known / len(rows) < 0.95:
            print("FruQ-DB file names don't identify the fruit -> using FruQ-multi (4.7 GB).")
            rows = []
    if not rows:
        rows, unk = index_images(download(CFG["multi_zip"]))
        print(f"FruQ-multi: {len(rows)} labelled images ({unk} without class)")
    rows = [r for r in rows if r["fruit"] is not None]
    per = {}
    for r in rows:
        per.setdefault(r["fruit"], Counter())[r["label"]] += 1
    print("\nImages per fruit (Fresh / Mild / Rotten):")
    for fr in sorted(per):
        c = per[fr]
        print(f"  {fr:12s} {c[0]:5d} {c[1]:5d} {c[2]:5d}")
    good = sorted(fr for fr, c in per.items() if all(c[k] > 0 for k in range(3)))
    rows = sorted([r for r in rows if r["fruit"] in good], key=lambda r: r["path"])
    return rows, good


def load_cache(rows):
    S = CFG["cache_size"]
    def load(p):
        return np.asarray(Image.open(p).convert("RGB").resize((S, S), Image.BILINEAR))
    print(f"Loading {len(rows)} images into memory at {S}x{S} ...")
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(load, [r["path"] for r in rows])))
    print(f"  cache: {X.nbytes / 1e9:.2f} GB")
    return X


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
                      num_workers=4, pin_memory=True, persistent_workers=False)


# ----------------------------------------------------------- folds
def make_folds(rows, fruits):
    rng = random.Random(CFG["split_seed"])
    order = fruits[:]; rng.shuffle(order)
    fruit_arr = np.array([r["fruit"] for r in rows])
    label_arr = np.array([r["label"] for r in rows])
    folds = []
    for i, test_f in enumerate(order):
        calib_f = order[(i + 1) % len(order)]
        train_f = [f for f in order if f not in (test_f, calib_f)]
        tr_all = np.where(np.isin(fruit_arr, train_f))[0]
        strat = [f"{fruit_arr[j]}_{label_arr[j]}" for j in tr_all]
        tr, va = train_test_split(tr_all, test_size=0.15, stratify=strat,
                                  random_state=CFG["split_seed"])
        folds.append(dict(fold=i, test=test_f, calib=calib_f, train=train_f, tr=tr, va=va,
                          ca=np.where(fruit_arr == calib_f)[0],
                          te=np.where(fruit_arr == test_f)[0]))
    return folds


# -------------------------------------------------- shared evaluation
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
    out = [(c @ bank.T).topk(CFG["knn_k"], dim=1).values[:, -1] for c in hn.split(1024)]
    return torch.cat(out)


def make_bank(h_tr):
    idx = torch.randperm(len(h_tr), generator=torch.Generator().manual_seed(0))[:CFG["max_bank"]]
    return F.normalize(h_tr[idx], dim=1)


def rejection_metrics(score, correct):
    score, correct = score.numpy(), correct.numpy().astype(float)
    c_sorted = correct[np.argsort(-score)]
    k = int(round(len(c_sorted) * 0.8))
    return dict(
        auroc=roc_auc_score(correct, score) if 0 < correct.mean() < 1 else float("nan"),
        aurc=float((np.cumsum(1 - c_sorted) / np.arange(1, len(c_sorted) + 1)).mean()),
        acc_rej20=float(c_sorted[:k].mean()),
    )


def evaluate(lva, yva, lca, yca, lte, yte, hte, bank):
    T_in, T_un = fit_temperature(lva, yva), fit_temperature(lca, yca)
    pred = lte.argmax(1)
    correct = pred == yte
    y, p = yte.numpy(), pred.numpy()
    res = dict(
        T_in=T_in, T_unseen=T_un,
        acc=correct.float().mean().item(),
        bal_acc=balanced_accuracy_score(y, p),
        macro_f1=f1_score(y, p, average="macro", labels=[0, 1, 2]),
        recall=recall_score(y, p, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
        in_domain_val_acc=(lva.argmax(1) == yva).float().mean().item(),
        ece={"none": ece_score(F.softmax(lte, 1), yte),
             "TS in-domain": ece_score(F.softmax(lte / T_in, 1), yte),
             "TS unseen-fruit": ece_score(F.softmax(lte / T_un, 1), yte)},
        rejection={"MSP": rejection_metrics(F.softmax(lte, 1).max(1).values, correct),
                   "kNN": rejection_metrics(knn_score(hte, bank), correct)},
    )
    return res


# ------------------------------------------------------- DINOv2 probe
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
    for x, _ in loader(X, np.zeros(len(X), int), np.arange(len(X)), False, bs=128):
        with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
            feats.append(m(x.to(DEV)).float().cpu())
    del m
    torch.cuda.empty_cache()
    return torch.cat(feats)


def run_dino(X, Y, fruit_arr, folds):
    path = f"{CFG['out']}/dino_features.npz"
    if os.path.exists(path):
        Fe = torch.from_numpy(np.load(path)["features"])
    else:
        Fe = dino_features(X)
        np.savez_compressed(path, features=Fe.numpy(), labels=Y, fruits=fruit_arr)
    print(f"DINOv2 features: {tuple(Fe.shape)}")
    Yt = torch.from_numpy(Y)
    out = {}
    for variant, cw in [("DINOv2-LP", None), ("DINOv2-LP-balanced", "balanced")]:
        runs = []
        for fd in folds:
            sc = StandardScaler().fit(Fe[fd["tr"]].numpy())
            clf = LogisticRegression(C=CFG["dino_C"], max_iter=5000, class_weight=cw)
            clf.fit(sc.transform(Fe[fd["tr"]].numpy()), Y[fd["tr"]])
            lg = lambda ix: torch.tensor(clf.decision_function(sc.transform(Fe[ix].numpy())),
                                         dtype=torch.float32)
            lte = lg(fd["te"])
            r = evaluate(lg(fd["va"]), Yt[fd["va"]], lg(fd["ca"]), Yt[fd["ca"]],
                         lte, Yt[fd["te"]], Fe[fd["te"]], make_bank(Fe[fd["tr"]]))
            r.update(model=variant, seed=0, fold=fd["fold"], test=fd["test"], calib=fd["calib"])
            runs.append((r, lte))
            print(f"  {variant:20s} test {fd['test']:10s} bal acc {r['bal_acc']:.3f} "
                  f"recall {np.round(r['recall'], 2)}")
        out[variant] = runs
    return out


# --------------------------------------------------- MobileNetV3 (FT)
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
    H, L, Ys = [], [], []
    for x, y in ld:
        h, lg = forward_feats(model, x.to(DEV))
        H.append(h.float().cpu()); L.append(lg.float().cpu()); Ys.append(y)
    return torch.cat(H), torch.cat(L), torch.cat(Ys)


def train_one(model, tr_loader, va_loader):
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))
    best, best_state = 1e9, None
    for ep in range(CFG["epochs"]):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(DEV == "cuda")):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        _, lg, yv = run_model(model, va_loader)
        vloss = F.cross_entropy(lg, yv).item()
        print(f"    epoch {ep+1:02d}  in-domain val NLL {vloss:.4f}  "
              f"acc {(lg.argmax(1) == yv).float().mean():.4f}")
        if vloss < best:
            best, best_state = vloss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def run_mobilenet_fold(X, Y, fd, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = train_one(build_model(3), loader(X, Y, fd["tr"], True), loader(X, Y, fd["va"], False))
    if CFG["save_weights"]:
        torch.save(model.state_dict(), f"{CFG['out']}/mnv3_f{fd['fold']}_s{seed}.pt")
    bank_idx = np.random.RandomState(0).permutation(fd["tr"])[:CFG["max_bank"]]
    htr, _, _ = run_model(model, loader(X, Y, bank_idx, False))
    _, lva, yva = run_model(model, loader(X, Y, fd["va"], False))
    _, lca, yca = run_model(model, loader(X, Y, fd["ca"], False))
    hte, lte, yte = run_model(model, loader(X, Y, fd["te"], False))
    r = evaluate(lva, yva, lca, yca, lte, yte, hte, F.normalize(htr, dim=1))
    r.update(model="MobileNetV3-FT", seed=seed, fold=fd["fold"], test=fd["test"], calib=fd["calib"])
    return r, lte


# ------------------------------------------------------------- summary
def pooled(runs, Y, folds):
    """runs: list of (result, test_logits) across folds for one model+seed."""
    y = torch.cat([torch.from_numpy(Y[folds[r["fold"]]["te"]]) for r, _ in runs])
    L = torch.cat([lg for _, lg in runs])
    P_in = torch.cat([F.softmax(lg / r["T_in"], 1) for r, lg in runs])
    P_un = torch.cat([F.softmax(lg / r["T_unseen"], 1) for r, lg in runs])
    p = L.argmax(1).numpy()
    ms = lambda k, s: np.nanmean([r["rejection"][s][k] for r, _ in runs])
    return dict(
        n=len(y), acc=float((p == y.numpy()).mean()),
        bal_acc=balanced_accuracy_score(y.numpy(), p),
        macro_f1=f1_score(y.numpy(), p, average="macro"),
        recall=recall_score(y.numpy(), p, average=None, labels=[0, 1, 2], zero_division=0),
        cm=confusion_matrix(y.numpy(), p, labels=[0, 1, 2]),
        ece_none=ece_score(F.softmax(L, 1), y), ece_in=ece_score(P_in, y), ece_un=ece_score(P_un, y),
        un_wins=sum(r["ece"]["TS unseen-fruit"] < r["ece"]["TS in-domain"] for r, _ in runs),
        auroc_msp=ms("auroc", "MSP"), auroc_knn=ms("auroc", "kNN"),
        rej_msp=ms("acc_rej20", "MSP"), rej_knn=ms("acc_rej20", "kNN"),
        knn_wins=sum(r["rejection"]["kNN"]["aurc"] < r["rejection"]["MSP"]["aurc"] for r, _ in runs),
    )


def write_summary(all_runs, Y, folds):
    L = [f"FruQ leave-one-fruit-out | {len(folds)} folds | 7 train fruits, 1 calib, 1 test",
         PILOT3_REF, "",
         "POOLED over all test fruits (each fruit tested once):",
         f"{'model':26s}{'folds':>6s}{'bal_acc':>9s}{'mF1':>7s}{'acc':>7s}{'rec F/M/R':>18s}"
         f"{'ECE none/in/unseen':>22s}{'unseen>in':>10s}{'AUROC MSP/kNN':>15s}"
         f"{'acc@rej20 MSP/kNN':>19s}{'kNN>MSP':>8s}"]
    per_fruit = []
    for name, runs in all_runs.items():
        if not runs:
            continue
        s = pooled(runs, Y, folds)
        k = len(runs)
        rec = "/".join(f"{v:.2f}" for v in s["recall"])
        un_w, knn_w = f"{s['un_wins']}/{k}", f"{s['knn_wins']}/{k}"
        L.append(f"{name:26s}{k:>6d}{s['bal_acc']:>9.3f}{s['macro_f1']:>7.3f}{s['acc']:>7.3f}"
                 f"{rec:>18s}{s['ece_none']:>8.3f}/{s['ece_in']:.3f}/{s['ece_un']:.3f}"
                 f"{un_w:>10s}{s['auroc_msp']:>9.3f}/{s['auroc_knn']:.3f}"
                 f"{s['rej_msp']:>13.3f}/{s['rej_knn']:.3f}{knn_w:>8s}")
        per_fruit.append(f"\n{name}: confusion (rows true F/M/R, cols predicted)\n{s['cm']}")
        per_fruit.append(f"  {'test fruit':12s}{'acc':>7s}{'bal_acc':>9s}{'rec F/M/R':>18s}"
                         f"{'ECE in/unseen':>16s}{'T in/unseen':>14s}{'in-dom val':>11s}")
        for r, _ in runs:
            rec = "/".join(f"{v:.2f}" for v in r["recall"])
            per_fruit.append(f"  {r['test']:12s}{r['acc']:>7.3f}{r['bal_acc']:>9.3f}{rec:>18s}"
                             f"{r['ece']['TS in-domain']:>9.3f}/{r['ece']['TS unseen-fruit']:<6.3f}"
                             f"{r['T_in']:>8.2f}/{r['T_unseen']:<7.2f}{r['in_domain_val_acc']:>9.3f}")
    L += ["", "bal_acc = mean of per-class recall (chance = 0.333). 'unseen>in' = folds where",
          "calibrating on the held-out fruit gave lower ECE than calibrating on in-domain val.",
          "'kNN>MSP' = folds where kNN ranked errors better (lower AURC)."]
    txt = "\n".join(L + per_fruit)
    open(f"{CFG['out']}/summary.txt", "w").write(txt)
    json.dump({n: [r for r, _ in rs] for n, rs in all_runs.items()},
              open(f"{CFG['out']}/results.json", "w"), indent=1)
    return txt


# ---------------------------------------------------------------- main
def main():
    rows, fruits = load_index()
    Y = np.array([r["label"] for r in rows])
    fruit_arr = np.array([r["fruit"] for r in rows])
    folds = make_folds(rows, fruits)
    json.dump([dict(fold=f["fold"], test=f["test"], calib=f["calib"], train=f["train"],
                    n_train=len(f["tr"]), n_val=len(f["va"]), n_calib=len(f["ca"]), n_test=len(f["te"]))
               for f in folds], open(f"{CFG['out']}/folds.json", "w"), indent=1)
    for f in folds:
        print(f"fold {f['fold']}: test {f['test']:10s} calib {f['calib']:10s} "
              f"n train/val/calib/test {len(f['tr'])}/{len(f['va'])}/{len(f['ca'])}/{len(f['te'])}")
    X = load_cache(rows)

    all_runs = {}
    if CFG["run_dino"]:
        print("\n##### DINOv2 linear probe")
        all_runs.update(run_dino(X, Y, fruit_arr, folds))
        print("\n" + write_summary(all_runs, Y, folds))

    if CFG["run_mobilenet"]:
        for seed in CFG["mobilenet_seeds"]:
            name = f"MobileNetV3-FT seed{seed}"
            all_runs[name] = []
            for fd in folds:
                print(f"\n##### {name} | fold {fd['fold']} test {fd['test']} calib {fd['calib']}")
                r, lte = run_mobilenet_fold(X, Y, fd, seed)
                all_runs[name].append((r, lte))
                np.savez_compressed(f"{CFG['out']}/mnv3_f{fd['fold']}_s{seed}.npz",
                                    logits=lte.numpy(), labels=Y[fd["te"]])
                print(f"  bal acc {r['bal_acc']:.3f} recall {np.round(r['recall'], 2)}")
                write_summary(all_runs, Y, folds)
    print("\n" + write_summary(all_runs, Y, folds))


if __name__ == "__main__":
    main()
