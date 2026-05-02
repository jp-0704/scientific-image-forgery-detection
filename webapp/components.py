"""
Streamlit UI components for the forgery detection webapp.

Public functions:
    upload_single_image()       -> np.ndarray RGB or None
    upload_batch()              -> list of (filename, np.ndarray RGB) or None
    render_prediction(result, original_rgb)
    render_comparison_table(results: dict[model_id, result])
    model_selector(default=None) -> str (model_id)
    render_performance_dashboard()
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from inference import (discover_models, MODEL_REGISTRY, load_comparison_csv,
                        load_metrics_json, model_artefact_dir)


# ── Image decoding helpers ───────────────────────────────────────────────────

_VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def _decode_bytes_to_rgb(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Upload widgets ───────────────────────────────────────────────────────────

def upload_single_image(key: str = 'single_uploader'):
    """Returns (filename, rgb_array) or (None, None).

    Pass a unique `key` per call site — Streamlit forbids duplicate widget keys
    across the whole app.
    """
    f = st.file_uploader(
        'Upload a single image',
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'],
        key=key,
    )
    if f is None:
        return None, None
    rgb = _decode_bytes_to_rgb(f.getvalue())
    if rgb is None:
        st.error(f'Failed to decode image: {f.name}')
        return None, None
    return f.name, rgb


def upload_batch(key_prefix: str = 'batch'):
    """
    Accept a zip OR multiple image files.
    Returns list of (filename, rgb_array) or empty list.

    `key_prefix` namespaces every internal widget so this can be used in
    multiple tabs without colliding.
    """
    mode = st.radio('Batch input mode',
                    ['Multiple image files', 'ZIP archive'],
                    horizontal=True, key=f'{key_prefix}_mode')
    out = []

    if mode == 'Multiple image files':
        files = st.file_uploader(
            'Upload images',
            type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'],
            accept_multiple_files=True,
            key=f'{key_prefix}_files',
        )
        if files:
            for f in files:
                rgb = _decode_bytes_to_rgb(f.getvalue())
                if rgb is not None:
                    out.append((f.name, rgb))
    else:
        f = st.file_uploader('Upload a ZIP archive', type=['zip'],
                             key=f'{key_prefix}_zip')
        if f is not None:
            try:
                with zipfile.ZipFile(io.BytesIO(f.getvalue())) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        ext = Path(info.filename).suffix.lower()
                        if ext not in _VALID_EXTS:
                            continue
                        with zf.open(info) as fp:
                            rgb = _decode_bytes_to_rgb(fp.read())
                            if rgb is not None:
                                out.append((info.filename, rgb))
            except zipfile.BadZipFile:
                st.error('Invalid ZIP file')
    return out


# ── Model selector ───────────────────────────────────────────────────────────

def model_selector(label: str = 'Choose model', default: str | None = None,
                   key: str = 'model_select'):
    """
    Returns (filename, threshold) or (None, None) if no models found.

    `threshold` reflects the user's choice — either the model's saved value or
    a slider override.
    """
    discovered = discover_models()
    if not discovered:
        st.warning('No model weights found in `webapp/models/`. '
                   'Drop any `.pt` file there — architecture is auto-detected.')
        return None, None

    by_filename = {m['filename']: m for m in discovered}
    options = list(by_filename.keys())
    idx = 0
    if default and default in options:
        idx = options.index(default)

    chosen = st.selectbox(
        label, options, index=idx,
        format_func=lambda fn: by_filename[fn]['display'],
        key=key,
    )

    info       = by_filename[chosen]
    arch_id    = info['arch_id']
    arch_desc  = MODEL_REGISTRY.get(arch_id, {}).get('description', '')
    meta       = info.get('meta', {})
    saved_thr  = info['threshold']
    thr_source = info['thr_source']

    # Caption: arch description + saved metrics
    bits = []
    if arch_desc:
        bits.append(arch_desc)
    if 'val_auc' in meta:
        try:
            bits.append(f"val AUC = {float(meta['val_auc']):.4f}")
        except (TypeError, ValueError):
            pass
    if 'val_dice' in meta:
        try:
            bits.append(f"val Dice = {float(meta['val_dice']):.4f}")
        except (TypeError, ValueError):
            pass
    if 'val_specificity' in meta:
        try:
            bits.append(f"val Spec = {float(meta['val_specificity']):.4f}")
        except (TypeError, ValueError):
            pass
    if bits:
        st.caption(' · '.join(bits))

    # Threshold display + override
    src_label = {
        'saved':        'from checkpoint',
        'arch default': f'architecture default for `{arch_id}`',
        'fallback':     'generic fallback (set in checkpoint or override below)',
    }.get(thr_source, thr_source)

    st.caption(f'Classification threshold for this model: '
               f'**{saved_thr:.2f}**  ({src_label})')

    use_override = st.checkbox(
        'Override threshold for this run',
        value=False,
        key=f'{key}_override_toggle',
    )
    if use_override:
        threshold = st.slider(
            'Custom classification threshold',
            min_value=0.05, max_value=0.95,
            value=float(saved_thr), step=0.01,
            key=f'{key}_override_slider',
        )
    else:
        threshold = saved_thr

    return chosen, float(threshold)


# ── Prediction display ───────────────────────────────────────────────────────

def render_prediction(result: dict, original_rgb: np.ndarray, filename: str = ''):
    """Show a single prediction with all panels."""
    label = result['label']
    prob  = result['cls_prob']
    thr   = result['threshold']
    mask  = result['mask']
    heat  = result['heatmap']
    pct   = result['forged_pct']

    # Headline
    if label == 'Forged':
        st.error(f'**{label.upper()}**  —  confidence {prob:.3f}  '
                 f'(threshold {thr:.2f})')
    else:
        st.success(f'**{label.upper()}**  —  confidence {1 - prob:.3f}  '
                   f'(threshold {thr:.2f})')

    cols = st.columns(3 if label == 'Forged' else 1)

    # Panel 1: input
    with cols[0]:
        st.image(original_rgb, caption=f'Input{": " + filename if filename else ""}',
                 width="stretch")

    if label == 'Forged':
        # Build red overlay
        overlay = original_rgb.copy().astype(np.float32)
        red = np.array([255.0, 40.0, 40.0])
        m = mask > 0
        overlay[m] = overlay[m] * 0.45 + red * 0.55
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        with cols[1]:
            st.image(overlay,
                     caption=f'Forged region overlay  ({pct:.1f}% of image)',
                     width="stretch")
        with cols[2]:
            st.image(heat,
                     caption='Segmentation heatmap (jet)',
                     width="stretch")

        st.caption(f'Inference device: {result["device"]}  |  '
                   f'TTA: {"on" if result["used_tta"] else "off"}')
    else:
        st.caption(f'Inference device: {result["device"]}  |  '
                   f'TTA: {"on" if result["used_tta"] else "off"}')


# ── Batch results table ──────────────────────────────────────────────────────

def render_batch_table(rows: list, csv_name: str = 'predictions.csv'):
    """rows: list of dicts with at least filename/label/cls_prob/forged_pct."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

    csv_buf = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download predictions as CSV',
        data=csv_buf,
        file_name=csv_name,
        mime='text/csv',
    )

    if 'label' in df.columns:
        n_total = len(df)
        n_forg  = int((df['label'] == 'Forged').sum())
        n_auth  = int((df['label'] == 'Authentic').sum())
        c1, c2, c3 = st.columns(3)
        c1.metric('Total', n_total)
        c2.metric('Forged', n_forg)
        c3.metric('Authentic', n_auth)


# ── Multi-model comparison ───────────────────────────────────────────────────

def render_comparison_grid(results: dict, original_rgb: np.ndarray,
                           displays: dict | None = None):
    """results: { filename: result_dict }.  displays: { filename: display_str }."""
    if not results:
        return
    displays = displays or {}
    n = len(results)
    cols = st.columns(n)
    for col, (fn, result) in zip(cols, results.items()):
        with col:
            display = displays.get(fn, fn)
            st.markdown(f'**{display}**')
            label = result['label']
            prob  = result['cls_prob']
            badge = ('🔴' if label == 'Forged' else '🟢')
            st.markdown(f'{badge} {label}  —  p={prob:.3f}')
            if label == 'Forged':
                overlay = original_rgb.copy().astype(np.float32)
                m = result['mask'] > 0
                overlay[m] = overlay[m] * 0.45 + np.array([255.0, 40.0, 40.0]) * 0.55
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                st.image(overlay, width="stretch")
            else:
                st.image(original_rgb, width="stretch")


# ── Performance dashboard ────────────────────────────────────────────────────

def render_performance_dashboard():
    """
    Two-section dashboard:
      A. Cross-model comparison (from compare_models.ipynb)
      B. Per-model artefacts (training curves, dashboard, predictions, metrics card)
         pulled from outputs/<train_X_arch_id>/ for every architecture currently
         loaded in webapp/models/.
    """
    # ── A. Cross-model comparison ────────────────────────────────────────────
    st.markdown('## Cross-Model Comparison')
    df = load_comparison_csv()
    if df is not None:
        st.dataframe(df, width="stretch")
    else:
        st.info('No `outputs/comparison/comparison.csv` yet. '
                'Run `compare_models.ipynb` to generate cross-model artefacts.')

    base_outputs = Path(__file__).parent.parent / 'outputs' / 'comparison'
    dash = base_outputs / 'comparison_dashboard.png'
    roc  = base_outputs / 'roc_overlay.png'
    if dash.exists():
        st.markdown('### Comparison dashboard')
        st.image(str(dash), width="stretch")
    if roc.exists():
        st.markdown('### ROC overlay')
        st.image(str(roc), width="stretch")

    # Show winner card if winner.json exists
    winner_path = base_outputs / 'winner.json'
    if winner_path.exists():
        try:
            import json as _json
            with open(winner_path) as f:
                w = _json.load(f)
            st.success(f"🏆 **Best model by composite metric: "
                       f"{w.get('display', w.get('model'))}** — composite = "
                       f"{w.get('composite', 0):.4f}")
        except Exception:
            pass

    # ── B. Per-model artefacts ───────────────────────────────────────────────
    st.markdown('---')
    st.markdown('## Per-Model Training Artefacts')

    discovered = discover_models()
    if not discovered:
        st.info('No models loaded; nothing to render.')
        return

    # Group by arch_id so we don't duplicate when multiple .pt files share an arch
    seen_archs = set()
    for m in discovered:
        arch_id = m['arch_id']
        if arch_id in seen_archs:
            continue
        seen_archs.add(arch_id)

        with st.expander(f"📊 {m['display']}  —  arch_id `{arch_id}`", expanded=False):
            adir = model_artefact_dir(arch_id)
            if adir is None:
                st.info(f'No artefacts found under `outputs/` for `{arch_id}`. '
                        'Run the corresponding training notebook to generate them.')
                continue

            st.caption(f'Artefacts directory: `{adir.relative_to(Path(__file__).parent.parent)}`')

            # Metrics card
            metrics, _ = load_metrics_json(arch_id)
            if metrics:
                cols = st.columns(4)
                cols[0].metric('AUC',         f"{metrics.get('auc', 0):.4f}")
                cols[1].metric('F1',          f"{metrics.get('f1', 0):.4f}")
                cols[2].metric('Specificity', f"{metrics.get('specificity', 0):.4f}")
                cols[3].metric('Mean Dice',   f"{metrics.get('mean_dice', 0):.4f}")
                cols2 = st.columns(4)
                cols2[0].metric('Forged Recall', f"{metrics.get('forged_recall', 0):.4f}")
                cols2[1].metric('Precision',    f"{metrics.get('precision', 0):.4f}")
                cols2[2].metric('Threshold',    f"{metrics.get('threshold', 0):.2f}")
                cols2[3].metric('Composite',    f"{metrics.get('composite', 0):.4f}")

            # Training curves
            tc = adir / 'training_curves.png'
            if tc.exists():
                st.markdown('**Training curves**')
                st.image(str(tc), width="stretch")

            # Dashboard (score distribution + CM + ROC)
            d = adir / 'dashboard.png'
            if d.exists():
                st.markdown('**Validation dashboard** (score distribution · confusion matrix · ROC)')
                st.image(str(d), width="stretch")

            # Predictions grid
            p = adir / 'predictions.png'
            if p.exists():
                st.markdown('**Qualitative predictions** (forged val examples)')
                st.image(str(p), width="stretch")

            # Authentic FPs (baseline only)
            afp = adir / 'authentic_false_positives.png'
            if afp.exists():
                st.markdown('**Authentic images sorted by forged-probability**')
                st.image(str(afp), width="stretch")

    if df is None and not list(seen_archs):
        st.warning('No artefacts available. Run the training notebooks first.')
