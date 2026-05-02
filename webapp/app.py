"""
Streamlit entry point for the forgery detection webapp.

Run:
    cd webapp && streamlit run app.py

Tabs:
    1. Single Image     drag-drop one image -> label, confidence, heatmap, mask
    2. Batch / Folder   upload zip or many images -> CSV + summary
    3. Compare Models   one image, all available models side-by-side
    4. Performance      static dashboard from outputs/comparison/
"""

from __future__ import annotations

import streamlit as st
import torch

from components import (
    upload_single_image,
    upload_batch,
    model_selector,
    render_prediction,
    render_batch_table,
    render_comparison_grid,
    render_performance_dashboard,
)
from inference import (
    predict_image,
    discover_models,
    MODEL_REGISTRY,
)


st.set_page_config(
    page_title='Scientific Image Forgery Detection',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title('🔬 Forgery Detection')
st.sidebar.markdown(
    'Upload a scientific image to classify it as **authentic** or **forged**, '
    'and visualise the suspected copy-move region.'
)

device = 'CUDA' if torch.cuda.is_available() else 'CPU'
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'
st.sidebar.info(f'**Inference device:** {device}\n\n{gpu_name}')

discovered = discover_models()
n = len(discovered)
st.sidebar.success(f'**{n} model(s) loaded**')
for m in discovered:
    st.sidebar.markdown(f"- `{m['filename']}` — {m['display']}")

with st.sidebar.expander('Inference settings'):
    use_tta = st.checkbox('Use 4-fold TTA (slower, slightly more accurate)',
                          value=True)
    min_area = st.slider('Min component area (pixels)', 0, 1000, 200, 50)


# ── Main tabs ────────────────────────────────────────────────────────────────

tab_single, tab_batch, tab_compare, tab_perf = st.tabs([
    '🖼️  Single Image',
    '📂 Batch / Folder',
    '⚖️  Compare Models',
    '📊 Performance',
])


# ── Tab 1: Single image ──────────────────────────────────────────────────────

with tab_single:
    st.header('Single Image Prediction')

    chosen_model, chosen_thr = model_selector('Choose model', key='single_model_select')

    fname, rgb = upload_single_image(key='single_uploader')
    if rgb is not None and chosen_model:
        if st.button('Run prediction', type='primary', key='single_btn'):
            with st.spinner('Predicting...'):
                result = predict_image(rgb, chosen_model,
                                       use_tta=use_tta,
                                       cls_thr=chosen_thr,
                                       seg_thr=chosen_thr,
                                       min_area=min_area)
            render_prediction(result, rgb, filename=fname or '')

            with st.expander('Raw output'):
                st.json({
                    'label':      result['label'],
                    'cls_prob':   result['cls_prob'],
                    'threshold':  result['threshold'],
                    'forged_pct': result['forged_pct'],
                    'device':     result['device'],
                    'used_tta':   result['used_tta'],
                })


# ── Tab 2: Batch ─────────────────────────────────────────────────────────────

with tab_batch:
    st.header('Batch Prediction')
    st.caption('Upload a folder (as ZIP) or multiple files. '
               'Images are processed sequentially; CSV download appears below.')

    chosen_model, chosen_thr = model_selector('Choose model', key='batch_model_select')

    items = upload_batch(key_prefix='batch')
    if items and chosen_model:
        if st.button(f'Run on {len(items)} image(s)', type='primary',
                     key='batch_btn'):
            progress = st.progress(0.0, text='Predicting...')
            rows = []
            for i, (name, img_rgb) in enumerate(items):
                try:
                    res = predict_image(img_rgb, chosen_model,
                                        use_tta=use_tta,
                                        cls_thr=chosen_thr,
                                        seg_thr=chosen_thr,
                                        min_area=min_area)
                    rows.append({
                        'filename':   name,
                        'label':      res['label'],
                        'cls_prob':   round(res['cls_prob'], 4),
                        'threshold':  res['threshold'],
                        'forged_pct': round(res['forged_pct'], 2),
                    })
                except Exception as e:
                    rows.append({
                        'filename': name,
                        'label':    'ERROR',
                        'cls_prob': 0.0,
                        'threshold': 0.0,
                        'forged_pct': 0.0,
                        'error':    str(e),
                    })
                progress.progress((i + 1) / len(items),
                                  text=f'Predicted {i + 1}/{len(items)}')
            progress.empty()
            st.success(f'Done. {len(rows)} predictions.')
            render_batch_table(rows, csv_name=f'{chosen_model}_predictions.csv')


# ── Tab 3: Compare models ────────────────────────────────────────────────────

with tab_compare:
    st.header('Side-by-side Model Comparison')
    st.caption('Same image, same TTA setting, every available model.')

    fname, rgb = upload_single_image(key='compare_uploader')
    if rgb is not None:
        st.caption('Each model uses its own per-architecture threshold (saved in '
                   'the checkpoint or arch-default).')
        if st.button('Run all models', type='primary', key='compare_btn'):
            results = {}
            displays = {}
            for m in discover_models():
                fn = m['filename']
                displays[fn] = f"{m['display']}  (thr={m['threshold']:.2f})"
                with st.spinner(f"Predicting with {m['display']}..."):
                    try:
                        # Use each model's own threshold (saved or arch default)
                        results[fn] = predict_image(
                            rgb, fn,
                            use_tta=use_tta,
                            cls_thr=m['threshold'],
                            seg_thr=m['threshold'],
                            min_area=min_area)
                    except Exception as e:
                        st.error(f"{fn}: {e}")
            if results:
                render_comparison_grid(results, rgb, displays=displays)


# ── Tab 4: Performance dashboard ─────────────────────────────────────────────

with tab_perf:
    st.header('Validation Performance Dashboard')
    st.caption('Generated by `compare_models.ipynb` on the held-out val set.')
    render_performance_dashboard()
