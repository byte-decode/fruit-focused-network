# Looking at the Fruit, Not the Background

Code, split definitions and result files for the paper:

> **Looking at the Fruit, Not the Background: Shortcut-Aware Evaluation and a Fruit-Focused Network for Image-Based Fruit Quality Grading**
> Rohit Agarwal (ByteDecode). *Submitted to Applied Fruit Science.*

The repository contains two parts:

1. **A shortcut audit** of three public fruit-quality datasets (FruQ, FruitNet, Fresh-Rotten): near-duplicate photographs, and how much of the label can be predicted from the photo background alone.
2. **The Fruit-Focused Network (FFN)**: MobileNetV3-Large with mask-supervised fruit-attention pooling and background-swap augmentation. Masks are used only during training; at inference the model takes an ordinary photograph.

## Main results

| | Baseline MobileNetV3 | FFN |
|---|---|---|
| FruitNet, honest split, natural photos | 0.997 | 0.996 |
| FruitNet, honest split, background swapped | 0.751 | **0.990** |
| FruitNet, unseen species, prediction flip rate when only the background changes | 26.0% | **12.1%** |
| Trained on FruitNet, tested on real Fresh-Rotten photos (3 seeds) | 0.839 ± 0.035 | **0.912 ± 0.004** |

Balanced accuracy unless stated. See the paper for all tables and limitations.

## Repository layout

```
scripts/
  01_fruq_leave_one_fruit_out.py         FruQ: leave-one-fruit-out (video-frame audit)
  02_fruitnet_audit_and_lofo.py          FruitNet: near-duplicate audit, random vs honest split, LOFO
  03_shortcut_square_masks.py            Shortcut probes with central-square masks and border colour
  04_shortcut_fruit_masks.py             Shortcut probes with CLIPSeg fruit masks; MobileNetV3 full vs fruit-only
  05_replicate_seeds_and_freshrotten.py  Extra seeds (FruitNet) and the same probes on Fresh-Rotten
  06_ffn_fruitnet.py                     FFN vs baseline + ablations (AP-only, BS-only), background-swap test
  07_ffn_seeds_and_crossdataset.py       FFN extra seeds, cross-dataset test, Fresh-Rotten LOFO
  analysis/fruq_block_consistency.py     FruQ: prediction consistency within fruit/stage blocks
  analysis/make_figures.py               Result charts used in the paper
splits/
  fruitnet_near_duplicate_groups.csv     Near-duplicate group of every FruitNet good/bad image
results/                                 summary.txt / results.json produced by each script
```

### Which script produced which result in the paper

| Paper | Script | Result folder |
|---|---|---|
| Table 1, near-duplicate audit (FruitNet) | 02 | `results/02_fruitnet_audit` |
| Table 1, near-duplicate audit (Fresh-Rotten) | 05 | `results/05_freshrotten` |
| Sect. 4.1, FruQ | 01, analysis/fruq_block_consistency.py | `results/01_fruq_lofo` |
| Sect. 4.2, random vs honest split | 02 | `results/02_fruitnet_audit` |
| Table 2 / Fig. 3, shortcut probes | 03, 04, 05 | `results/03_*`, `results/04_*`, `results/05_freshrotten` |
| Sect. 4.4, full vs fruit-only (3 seeds) | 04 (seed 0), 05 (seeds 1, 2) | `results/04_*`, `results/05_fruitnet_seeds_1_2` |
| Table 3, FFN in-domain + ablations | 06 | `results/06_ffn_fruitnet_seed0` |
| Table 4, FFN leave-one-fruit-out (3 seeds) | 06 (seed 0), 07 (seeds 1, 2) | `results/06_*`, `results/07_ffn_fruitnet_seeds_1_2` |
| Table 5 / Fig. 5, cross-dataset | 07 | `results/07_ffn_crossdataset_and_freshrotten` |
| Sect. 6.6, Fresh-Rotten LOFO | 07 | `results/07_ffn_crossdataset_and_freshrotten` |

## Data

The datasets are not redistributed here. Download them from the original sources:

- **FruitNet** (Meshram & Patil, Data in Brief, 2022): Kaggle `shashwatwork/fruitnet-indian-fruits-dataset-with-quality`
- **Fresh-Rotten**: Kaggle `sriramr/fruits-fresh-and-rotten-for-classification`
- **FruQ-DB**: Zenodo record 7224690 (downloaded automatically by script 01)

`splits/fruitnet_near_duplicate_groups.csv` lists, for each FruitNet good/bad image, its path inside the dataset, fruit, quality, near-duplicate group and the cosine similarity to its nearest neighbour (DINOv2-small). Images in the same group must stay on the same side of any train/test split. All scripts regenerate the groups and splits deterministically (seed 42).

## How to run

The experiments were run as Kaggle notebooks (NVIDIA T4 GPU, Internet on, because pretrained weights for MobileNetV3, DINOv2 and CLIPSeg are downloaded). Each script is self-contained:

1. Create a Kaggle notebook, add the dataset(s) named in the script's header as input, enable a GPU and Internet.
2. Upload the script and run `!python -u <script>.py`.
3. Outputs, including timestamped progress lines in `progress.txt`, are written to `/kaggle/working/<run name>/`.

Settings for each run (dataset, seeds, which variants or tasks to run) are in the `CFG` block at the top of each script; the header docstring explains the options. To run elsewhere, change `input_root` and the output path in `CFG`.

Approximate run times on one T4: script 06 about 3.5 h; script 07 about 3.5 h (cross-dataset + Fresh-Rotten) and 5 h (extra seeds); the others 10 min to 3 h.

## Requirements

Python 3.10+, PyTorch with torchvision, timm, transformers, scikit-learn, scipy, numpy, Pillow, matplotlib. See `requirements.txt`. The standard Kaggle GPU image already contains all of them.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences.
