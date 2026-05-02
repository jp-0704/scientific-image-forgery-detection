# Runbook — How to execute the project

Everything is built and ready. This file tells you exactly what to run, in what order, and where the artefacts land.

## Files produced this session

```
recod/
├── PLAN.txt                          # full plan + tables
├── RUNBOOK.md                        # this file
├── _build_notebooks.py               # template that generated the 3 train notebooks
│
├── train_1_convnext_tiny.ipynb       # Day 1 - upload to Kaggle, run
├── train_2_swin_v2_tiny.ipynb        # Day 2 - upload to Kaggle, run
├── train_3_swin_v2_base.ipynb        # Day 3 - upload to Kaggle, run (overnight)
├── compare_models.ipynb              # Day 4 - run after the 3 training notebooks
│
└── webapp/                           # Day 5 - run locally on RTX 3050 Ti
    ├── app.py                        # Streamlit entry (4 tabs)
    ├── arch.py                       # Shared model classes
    ├── inference.py                  # FP16 + TTA + OOM-safe loader
    ├── components.py                 # UI helpers
    ├── requirements.txt
    ├── README.md                     # quick-start for this folder only
    └── models/                       # drop trained .pt files here
```

---

## Day 1 — ConvNeXt-Tiny

1. Upload `train_1_convnext_tiny.ipynb` to Kaggle (or use the VSCode-compatible URL).
2. Attach the dataset: `recod-ailuc-scientific-image-forgery-detection`.
3. Set accelerator to **GPU T4 x2**.
4. Run all cells.
5. Expected runtime: **3–4 h**.
6. Outputs land in `/kaggle/working/outputs/convnext_tiny/`:
   - `model.pt`, `metrics.json`, `history.csv`
   - `dashboard.png`, `predictions.png`, `training_curves.png`
7. **Download `model.pt` and rename it to `convnext_tiny.pt`** when finished.

## Day 2 — Swin-V2-Tiny

Same procedure with `train_2_swin_v2_tiny.ipynb`. Expected runtime: **4–5 h**. Rename the saved `model.pt` to `swin_v2_tiny.pt`.

## Day 3 — Swin-V2-Base (heavy, can run overnight)

Same procedure with `train_3_swin_v2_base.ipynb`. Expected runtime: **8–12 h**. Has built-in OOM safeguards: gradient checkpointing on encoder, grad accumulation = 4, AMP. Rename the saved `model.pt` to `swin_v2_base.pt`.

If you hit `cuda.OutOfMemoryError`, edit cell 7 (Configuration) and reduce `BATCH_SIZE` to 2 with `GRAD_ACCUM = 8` (effective batch size stays 16).

## Day 4 — Comparison

Upload `compare_models.ipynb` to a new Kaggle session. Either:

- **Option A**: copy all 3 trained `model.pt` files to the right paths inside the kernel via a Kaggle dataset, OR
- **Option B**: download all 3 to your local machine and run the comparison locally with `jupyter lab compare_models.ipynb` (will take ~15 min on RTX 3050 Ti for inference only).

Outputs land in `outputs/comparison/`:
- `comparison.csv` (tabular metrics)
- `comparison_dashboard.png` (grouped bar charts)
- `roc_overlay.png` (4 ROC curves overlaid)
- `winner.json` (best model by composite score)

## Day 5 — Local Webapp

```bash
cd webapp
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Drop your trained weights into `webapp/models/` with these exact names:
```
webapp/models/
  efficientnet_b4.pt    # rename of final_model_v3.pt (the original baseline)
  convnext_tiny.pt      # from train_1
  swin_v2_tiny.pt       # from train_2
  swin_v2_base.pt       # from train_3
```

(The app auto-detects which models are present — works with just one or two.)

Run:
```bash
streamlit run app.py
```

Browser opens at `http://localhost:8501`.

---

## Recovery / troubleshooting

**Q: A training notebook crashed mid-run.**
A: Each notebook saves `outputs/<name>/best.pt` whenever the composite metric improves. Restart the kernel and re-run from cell 7 onwards if you want to resume from scratch, or load `best.pt` and continue manually.

**Q: Swin checkpoint shape mismatch on load.**
A: The provided `_to_nchw()` heuristic in `FourLevelUNet` handles both NCHW and NHWC features. If you see a mismatch, set `channels_last_input=True` (already done by default in the build_eval_model factory).

**Q: Webapp says "No model weights found".**
A: Confirm the filenames in `webapp/models/` are exactly:
- `efficientnet_b4.pt`, `convnext_tiny.pt`, `swin_v2_tiny.pt`, `swin_v2_base.pt`

**Q: RTX 3050 Ti runs out of memory in webapp.**
A: Toggle "Use 4-fold TTA" off in the sidebar. If still OOM, the app will auto-fall-back to CPU FP32. Inference will take ~2 s per image instead of 0.4 s.

---

## Validation summary

```
PLAN.txt              ✓ 11 sections including risk register
arch.py               ✓ compiles, defines 4 architecture wrappers
inference.py          ✓ compiles, OOM-safe, FP16, TTA fallback chain
components.py         ✓ compiles, batch + zip + CSV download
app.py                ✓ compiles, 4 tabs wired
train_1 notebook      ✓ 35 cells, all 17 code cells parse
train_2 notebook      ✓ 35 cells, all 17 code cells parse
train_3 notebook      ✓ 35 cells, all 17 code cells parse (incl. grad checkpointing)
compare notebook      ✓ 11 cells, all 5 code cells parse
```
