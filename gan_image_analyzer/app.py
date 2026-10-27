"""
GAN Image Quality and Diversity Analyzer
Main Streamlit Application
"""

import io
import os
import time
from typing import List, Dict, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

from model import (
    SUPPORTED_MODELS,
    load_gan_model,
    generate_gan_images,
    get_available_device
)
from preprocessing import (
    validate_and_load_image,
    extract_images_from_zip,
    package_generated_images
)
from analyzer import (
    analyze_dataset_quality,
    compute_pairwise_dhash_similarity,
    evaluate_diversity_report
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="GAN Image Quality & Diversity Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .badge-diverse {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-repetition {
        background-color: #fff3cd;
        color: #664d03;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-repetitive {
        background-color: #f8d7da;
        color: #842029;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .gallery-card {
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 8px;
        background: #ffffff;
        text-align: center;
        margin-bottom: 15px;
    }
    .subtext {
        font-size: 0.82rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "dataset" not in st.session_state:
    st.session_state["dataset"] = []

if "source_metadata" not in st.session_state:
    st.session_state["source_metadata"] = {
        "source": "None",
        "description": "No images loaded yet."
    }

if "cached_models" not in st.session_state:
    st.session_state["cached_models"] = {}


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("🔬 GAN Analyzer Controls")

device = get_available_device()
device_label = "CUDA GPU 🚀" if device.type == "cuda" else "CPU ⚙️"
st.sidebar.caption(f"**Execution Hardware:** {device_label}")

data_source = st.sidebar.radio(
    "Data Source Mode",
    ["⚡ Pretrained GAN Generation", "📁 Upload Images or ZIP", "🧪 Demo Sample Batch"],
    index=0
)

st.sidebar.markdown("---")

# 1. GAN Generation Controls
if data_source == "⚡ Pretrained GAN Generation":
    st.sidebar.subheader("Model Configuration")
    
    model_choices = {k: f"{v['name']}" for k, v in SUPPORTED_MODELS.items()}
    selected_model_key = st.sidebar.selectbox(
        "Select GAN Architecture",
        options=list(model_choices.keys()),
        format_func=lambda k: model_choices[k],
        index=0
    )
    
    model_info = SUPPORTED_MODELS[selected_model_key]
    st.sidebar.info(f"**Details:** {model_info['description']}\n\n**Resolution:** `{model_info['resolution']}` | **Latent:** `{model_info['latent_dim']}D`")
    
    num_gen = st.sidebar.slider("Number of Images to Generate", min_value=2, max_value=32, value=8, step=1)
    
    col_seed1, col_seed2 = st.sidebar.columns([3, 2])
    with col_seed1:
        seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)
    with col_seed2:
        if st.button("🎲 Randomize"):
            seed = int(np.random.randint(0, 999999))
            st.session_state["random_seed"] = seed

    if "random_seed" in st.session_state:
        seed = st.session_state["random_seed"]

    generate_btn = st.sidebar.button("🚀 Run GAN Inference", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner(f"Loading '{model_info['name']}' and running neural inference..."):
            try:
                # Load model
                if selected_model_key in st.session_state["cached_models"]:
                    loaded_model = st.session_state["cached_models"][selected_model_key]
                    err = None
                else:
                    loaded_model, _, err = load_gan_model(selected_model_key, device=device)
                    if loaded_model is not None:
                        st.session_state["cached_models"][selected_model_key] = loaded_model

                if err:
                    st.sidebar.error(f"Model Load Error: {err}")
                    st.error(f"**Model Loading Failed:** {err}\n\nYou can switch to another model (such as the self-contained DCGAN) or upload images manually.")
                else:
                    start_t = time.time()
                    raw_images = generate_gan_images(
                        loaded_model,
                        selected_model_key,
                        num_images=num_gen,
                        seed=int(seed),
                        device=device
                    )
                    elapsed = time.time() - start_t
                    
                    # Package with metadata and hashes
                    packaged_dataset = package_generated_images(
                        raw_images,
                        prefix=selected_model_key.lower().replace("-", "_")
                    )
                    
                    st.session_state["dataset"] = packaged_dataset
                    st.session_state["source_metadata"] = {
                        "source": f"GAN Generation ({model_info['name']})",
                        "model_key": selected_model_key,
                        "num_images": num_gen,
                        "seed": seed,
                        "elapsed_sec": round(elapsed, 2),
                        "resolution": model_info["resolution"]
                    }
                    st.sidebar.success(f"Generated {num_gen} images in {elapsed:.2f}s!")
            except Exception as e:
                st.sidebar.error(f"Inference failure: {e}")
                st.exception(e)

# 2. Upload Controls
elif data_source == "📁 Upload Images or ZIP":
    st.sidebar.subheader("Upload Images or Archives")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Image files or a ZIP archive",
        type=["png", "jpg", "jpeg", "webp", "bmp", "zip"],
        accept_multiple_files=True,
        help="Upload individual image files or a compressed .ZIP file containing images."
    )
    
    if uploaded_files:
        if st.sidebar.button("📥 Process Uploaded Files", type="primary", use_container_width=True):
            loaded_list = []
            with st.spinner("Validating and parsing uploaded images..."):
                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.read()
                    filename = uploaded_file.name
                    if filename.lower().endswith(".zip"):
                        extracted = extract_images_from_zip(file_bytes)
                        loaded_list.extend(extracted)
                    else:
                        item = validate_and_load_image(file_bytes, filename=filename)
                        if item:
                            loaded_list.append(item)
                
                if loaded_list:
                    st.session_state["dataset"] = loaded_list
                    st.session_state["source_metadata"] = {
                        "source": f"User Upload ({len(uploaded_files)} file(s))",
                        "num_images": len(loaded_list),
                        "seed": "N/A",
                        "elapsed_sec": "N/A"
                    }
                    st.sidebar.success(f"Loaded {len(loaded_list)} valid image(s)!")
                else:
                    st.sidebar.error("No valid images could be parsed from the uploaded files.")

# 3. Demo Sample Batch Controls
elif data_source == "🧪 Demo Sample Batch":
    st.sidebar.subheader("Sample Test Data")
    st.sidebar.write("Load a pre-generated sample batch of GAN images to quickly explore all metrics.")
    
    sample_dir = os.path.join(os.path.dirname(__file__), "samples")
    if st.sidebar.button("Load Pre-generated Sample Batch", type="primary", use_container_width=True):
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg'))]
            sample_images = []
            for sf in sorted(sample_files):
                full_path = os.path.join(sample_dir, sf)
                with open(full_path, "rb") as f:
                    bytes_data = f.read()
                parsed = validate_and_load_image(bytes_data, filename=sf)
                if parsed:
                    sample_images.append(parsed)
            if sample_images:
                st.session_state["dataset"] = sample_images
                st.session_state["source_metadata"] = {
                    "source": "Demo Pre-generated Sample Batch",
                    "num_images": len(sample_images),
                    "seed": "Sample Seed 42",
                    "elapsed_sec": "N/A"
                }
                st.sidebar.success(f"Loaded {len(sample_images)} demo sample images!")
        else:
            st.sidebar.warning("Sample directory not found.")


st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Analysis Parameters")
sim_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.50,
    max_value=0.98,
    value=0.85,
    step=0.01,
    help="Pairwise dHash similarity threshold above which image pairs are flagged as potential near-duplicates."
)

gallery_cols = st.sidebar.slider("Gallery Columns", min_value=2, max_value=6, value=4, step=1)

# Reset dataset button
if st.session_state["dataset"]:
    if st.sidebar.button("🗑️ Clear Active Dataset", use_container_width=True):
        st.session_state["dataset"] = []
        st.session_state["source_metadata"] = {"source": "None", "description": "Cleared."}
        st.rerun()


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("🔬 GAN Image Quality and Diversity Analyzer")
st.markdown(
    "Evaluate the visual quality, pixel statistics, structural uniqueness, "
    "and sample diversity of GAN-generated imagery without retraining models from scratch."
)

dataset = st.session_state["dataset"]
source_meta = st.session_state["source_metadata"]

if not dataset:
    st.info("👈 **Get Started:** Select a pretrained GAN model from the sidebar and click **'Run GAN Inference'**, or upload your own images/ZIP archives.")
    
    st.markdown("""
    ### Key Features
    - **Live PyTorch GAN Execution**: Supports Progressive GAN (PGAN) models and Deep Convolutional GAN (DCGAN) generators with seed-controlled reproducible latent sampling.
    - **Image Quality Diagnostics**: Computes dimensions, luminance (brightness), RMS contrast, exposure health, and spatial sharpness (discrete Laplacian variance).
    - **Difference Hashing (dHash)**: Computes 64-bit gradient difference perceptual hashes to detect identical structures and near-duplicates invariant to minor scaling or noise.
    - **Diversity & Uniqueness Assessment**: Generates pairwise similarity distributions, duplicate clusters, and an automatic heuristic classification.
    - **Comprehensive Reporting**: Interactive visual charts and downloadable CSV/Markdown reports.
    """)
    st.stop()


# ---------------------------------------------------------
# Run Analytics Backend
# ---------------------------------------------------------
with st.spinner("Analyzing image quality and pairwise diversity..."):
    df_quality = analyze_dataset_quality(dataset)
    sim_matrix, dist_matrix = compute_pairwise_dhash_similarity(dataset)
    diversity_rep = evaluate_diversity_report(
        dataset,
        sim_matrix,
        dist_matrix,
        similarity_threshold=sim_threshold
    )

# Source banner
st.caption(
    f"**Active Dataset:** {source_meta.get('source', 'Unknown')} | "
    f"**Total Samples:** `{len(dataset)}` | "
    f"**Random Seed:** `{source_meta.get('seed', 'N/A')}` | "
    f"**Inference Time:** `{source_meta.get('elapsed_sec', 'N/A')}s`"
)

# ---------------------------------------------------------
# Main Interface Tabs
# ---------------------------------------------------------
tab_gallery, tab_quality, tab_diversity, tab_reports = st.tabs([
    "🖼️ Image Gallery",
    "✨ Quality Metrics",
    "🌐 Diversity Analysis",
    "📊 Reports & Export"
])


# =========================================================
# TAB 1: Image Gallery
# =========================================================
with tab_gallery:
    st.subheader(f"Dataset Gallery ({len(dataset)} Images)")
    
    # Render gallery in responsive grid
    cols = st.columns(gallery_cols)
    for idx, item in enumerate(dataset):
        col = cols[idx % gallery_cols]
        with col:
            st.image(item["image"], use_container_width=True)
            st.caption(
                f"**#{idx + 1:02d}: {item['filename']}**\n\n"
                f"`{item['width']}x{item['height']}` | `{item['size_bytes']/1024:.1f} KB`\n\n"
                f"dHash: `{item['dhash']}`"
            )

    # Detailed Inspect expander
    with st.expander("🔍 Inspect Full-Resolution Sample"):
        inspect_idx = st.selectbox(
            "Select Image to Inspect",
            options=list(range(len(dataset))),
            format_func=lambda i: f"Image #{i+1:02d} ({dataset[i]['filename']})"
        )
        selected_item = dataset[inspect_idx]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(selected_item["image"], caption=f"Image #{inspect_idx + 1}", use_container_width=True)
        with c2:
            st.markdown(f"### Image #{inspect_idx + 1:02d} Metadata")
            st.write(f"- **Filename:** `{selected_item['filename']}`")
            st.write(f"- **Dimensions:** `{selected_item['width']} x {selected_item['height']}` (Aspect: `{selected_item['aspect_ratio']}`)")
            st.write(f"- **File Size:** `{selected_item['size_bytes'] / 1024:.2f} KB`")
            st.write(f"- **Format:** `{selected_item['format']}` (Mode: `{selected_item['original_mode']}`)")
            st.write(f"- **SHA-256 Hash:**")
            st.code(selected_item["sha256"], language="text")
            st.write(f"- **Difference Hash (dHash):**")
            st.code(selected_item["dhash"], language="text")


# =========================================================
# TAB 2: Image Quality Metrics
# =========================================================
with tab_quality:
    st.subheader("Image Quality Diagnostics")
    
    # KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        avg_bright = df_quality["brightness_mean"].mean()
        m1.metric("Avg. Brightness (Luma)", f"{avg_bright:.1f} / 255", help="Standard ITU-R 601-2 mean grayscale intensity.")
    with m2:
        avg_contrast = df_quality["rms_contrast"].mean()
        m2.metric("Avg. RMS Contrast", f"{avg_contrast:.1f}", help="Root Mean Square standard deviation of pixel intensities.")
    with m3:
        avg_sharpness = df_quality["sharpness_laplacian"].mean()
        m3.metric("Avg. Laplacian Sharpness", f"{avg_sharpness:.1f}", help="Variance of discrete 4-neighborhood Laplacian.")
    with m4:
        avg_sat = df_quality["mean_saturation"].mean()
        m4.metric("Avg. HSV Saturation", f"{avg_sat:.1f} / 255", help="Average color saturation in HSV space.")

    # Limitation Callout
    st.warning(
        "⚠️ **Diagnostic Limitations Notice:** Basic sharpness (Laplacian variance) and pixel statistics "
        "measure low-level spatial frequencies and luminance distributions. They **do not** fully capture "
        "semantic realism, perceptual plausibility, or characteristic GAN artifacts (e.g. checkerboard patterns, "
        "anatomical distortions, or unnatural high-frequency textures that can artificially inflate sharpness scores)."
    )

    # Detailed Quality Table
    st.markdown("### Per-Image Diagnostics Table")
    display_cols = [
        "index", "filename", "dimensions", "size_kb",
        "brightness_mean", "brightness_std", "under_exposed_pct", "over_exposed_pct",
        "rms_contrast", "sharpness_laplacian", "gradient_energy"
    ]
    st.dataframe(
        df_quality[display_cols].sort_values("index"),
        use_container_width=True,
        hide_index=True
    )

    # Charts Row
    st.markdown("### Quality Metric Distributions")
    q_col1, q_col2 = st.columns(2)
    
    with q_col1:
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
        ax_scatter.scatter(
            df_quality["brightness_mean"],
            df_quality["rms_contrast"],
            c=df_quality["sharpness_laplacian"],
            cmap="viridis",
            s=80,
            edgecolors="black",
            alpha=0.85
        )
        for _, row in df_quality.iterrows():
            ax_scatter.annotate(
                f"#{int(row['index'])}",
                (row["brightness_mean"], row["rms_contrast"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha='center',
                fontsize=8
            )
        ax_scatter.set_title("Brightness vs. RMS Contrast (Color = Sharpness)")
        ax_scatter.set_xlabel("Mean Brightness (0-255)")
        ax_scatter.set_ylabel("RMS Contrast")
        ax_scatter.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)

    with q_col2:
        fig_sharp, ax_sharp = plt.subplots(figsize=(6, 4))
        indices = df_quality["index"].astype(str)
        ax_sharp.bar(indices, df_quality["sharpness_laplacian"], color="#4e79a7", edgecolor="black")
        ax_sharp.axhline(avg_sharpness, color="red", linestyle="--", label=f"Mean: {avg_sharpness:.1f}")
        ax_sharp.set_title("Laplacian Sharpness by Sample")
        ax_sharp.set_xlabel("Image Index")
        ax_sharp.set_ylabel("Laplacian Variance")
        ax_sharp.legend()
        ax_sharp.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_sharp)
        plt.close(fig_sharp)


# =========================================================
# TAB 3: Diversity Analysis
# =========================================================
with tab_diversity:
    st.subheader("Image Diversity & Repetition Analysis")
    
    # Classification Badge Card
    badge_class = {
        "Appearing Diverse": "badge-diverse",
        "Showing Some Repetition": "badge-repetition",
        "Highly Repetitive": "badge-repetitive"
    }.get(diversity_rep["classification"], "badge-diverse")

    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin-top:0;">Diversity Assessment: <span class="{badge_class}">{diversity_rep['classification']}</span></h4>
        <p style="margin-bottom:4px;"><b>Analytical Rationale:</b> {diversity_rep['rationale']}</p>
        <p class="subtext">
            ⚠️ <b>Heuristic Disclaimer:</b> This assessment is a heuristic based on image hashing (dHash) and pairwise 
            Hamming distances within the generated batch. It does not provide definitive mathematical proof of GAN mode collapse, 
            which requires manifold-based divergence metrics (e.g. Fréchet Inception Distance or Precision/Recall) against the true dataset distribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Diversity KPI metrics
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        d1.metric(
            "Unique SHA-256 Hashes",
            f"{diversity_rep['unique_sha256_count']} / {diversity_rep['total_images']}",
            f"{diversity_rep['uniqueness_ratio_sha']*100:.1f}% Unique",
            help="Exact byte-level duplicate detection."
        )
    with d2:
        d2.metric(
            "Unique Perceptual Hashes",
            f"{diversity_rep['unique_dhash_count']} / {diversity_rep['total_images']}",
            f"{diversity_rep['uniqueness_ratio_dhash']*100:.1f}% Unique",
            help="Unique 64-bit difference hashes."
        )
    with d3:
        d3.metric(
            "Mean Pairwise Similarity",
            f"{diversity_rep['mean_similarity']:.3f}",
            f"Range: [{diversity_rep['min_similarity']:.2f}, {diversity_rep['max_similarity']:.2f}]",
            help="Average dHash similarity score across all unique pairs."
        )
    with d4:
        d4.metric(
            f"Pairs ≥ {sim_threshold:.2f} Threshold",
            f"{diversity_rep['near_duplicate_count']} / {diversity_rep['total_pairs']}",
            f"{diversity_rep['near_duplicate_pct']:.1f}% of pairs",
            delta_color="inverse",
            help="Image pairs exceeding the chosen similarity threshold."
        )

    # Visualizations
    st.markdown("### Pairwise Similarity Distributions")
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        # Histogram of pairwise similarities
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        pairwise_sims = diversity_rep["pairwise_similarities"]
        
        if len(pairwise_sims) > 0:
            bins = np.linspace(0.0, 1.0, 21)
            ax_hist.hist(pairwise_sims, bins=bins, color="#59a14f", edgecolor="black", alpha=0.8)
            ax_hist.axvline(sim_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({sim_threshold:.2f})")
            ax_hist.axvline(diversity_rep["mean_similarity"], color="blue", linestyle=":", linewidth=2, label=f"Mean ({diversity_rep['mean_similarity']:.2f})")
            ax_hist.set_title("Pairwise Similarity Distribution")
            ax_hist.set_xlabel("dHash Similarity Score (0.0 to 1.0)")
            ax_hist.set_ylabel("Pair Count")
            ax_hist.set_xlim(0.0, 1.0)
            ax_hist.legend()
            ax_hist.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with sim_col2:
        # Pairwise Similarity Heatmap Matrix
        fig_heat, ax_heat = plt.subplots(figsize=(6, 4))
        cax = ax_heat.imshow(sim_matrix, cmap="magma", vmin=0.0, vmax=1.0)
        fig_heat.colorbar(cax, ax=ax_heat, label="Similarity")
        
        n_img = len(dataset)
        ticks = np.arange(n_img)
        labels = [f"#{i+1}" for i in range(n_img)]
        ax_heat.set_xticks(ticks)
        ax_heat.set_yticks(ticks)
        ax_heat.set_xticklabels(labels, fontsize=8)
        ax_heat.set_yticklabels(labels, fontsize=8)
        ax_heat.set_title("Pairwise Similarity Heatmap")
        st.pyplot(fig_heat)
        plt.close(fig_heat)

    # Near-Duplicate / Most Similar Pairs Section
    st.markdown("### Most Similar & Potential Near-Duplicate Pairs")
    
    near_duplicates = diversity_rep["near_duplicate_pairs"]
    all_ranked = diversity_rep["all_pairs_ranked"]
    
    if near_duplicates:
        st.warning(f"Found **{len(near_duplicates)} pair(s)** exceeding similarity threshold `{sim_threshold:.2f}`.")
        pairs_to_show = near_duplicates[:6]
    else:
        st.success(f"No pairs exceeded the similarity threshold `{sim_threshold:.2f}`. Displaying top most-similar pairs for comparison:")
        pairs_to_show = all_ranked[:4]

    for p in pairs_to_show:
        with st.container():
            p_col1, p_col2, p_col3 = st.columns([2, 2, 3])
            with p_col1:
                st.image(p["img_a"], caption=f"#{p['idx_a']:02d}: {p['file_a']}", use_container_width=True)
            with p_col2:
                st.image(p["img_b"], caption=f"#{p['idx_b']:02d}: {p['file_b']}", use_container_width=True)
            with p_col3:
                st.markdown(f"#### Pair Comparison: #{p['idx_a']} vs #{p['idx_b']}")
                st.write(f"- **Similarity Score:** `{p['similarity']:.4f}`")
                st.write(f"- **Hamming Distance:** `{p['hamming_distance']} / 64 bits`")
                if p["is_exact_sha256"]:
                    st.error("🚨 **Exact Byte Duplicate:** Identical SHA-256 hash!")
                elif p["is_exact_dhash"]:
                    st.warning("⚠️ **Perceptually Identical:** 0 bit difference in dHash.")
                elif p["similarity"] >= sim_threshold:
                    st.info("⚠️ **Near-Duplicate:** High structural similarity.")
                else:
                    st.write("✅ **Visually Distinct**")
            st.divider()


# =========================================================
# TAB 4: Reports & Export
# =========================================================
with tab_reports:
    st.subheader("Automated Analysis Report & Data Exports")
    
    # Executive Summary Card
    summary_markdown = f"""
### Executive Summary
- **Evaluation Target:** `{source_meta.get('source', 'User Dataset')}`
- **Total Images Analyzed:** `{len(dataset)}`
- **Diversity Classification:** **{diversity_rep['classification']}**
- **Uniqueness Ratio (dHash):** `{diversity_rep['uniqueness_ratio_dhash']*100:.1f}%` ({diversity_rep['unique_dhash_count']} unique of {diversity_rep['total_images']})
- **Mean Pairwise Similarity:** `{diversity_rep['mean_similarity']:.3f}` (Median: `{diversity_rep['median_similarity']:.3f}`)
- **Potential Near-Duplicates:** `{diversity_rep['near_duplicate_count']}` pair(s) ({diversity_rep['near_duplicate_pct']:.1f}% of all pairs) above threshold `{sim_threshold:.2f}`
- **Quality Benchmarks:**
  - Average Brightness: `{avg_bright:.1f} / 255`
  - Average RMS Contrast: `{avg_contrast:.1f}`
  - Average Laplacian Sharpness: `{avg_sharpness:.1f}`
    """
    st.markdown(summary_markdown)

    st.markdown("### Download Data Reports")
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    
    # 1. Quality CSV
    with dl_col1:
        csv_quality = df_quality.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Quality Metrics CSV",
            data=csv_quality,
            file_name="gan_quality_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 2. Pairwise Similarity CSV
    with dl_col2:
        # Create pairwise dataframe
        pairwise_rows = []
        for p in diversity_rep["all_pairs_ranked"]:
            pairwise_rows.append({
                "pair_a_index": p["idx_a"],
                "pair_a_file": p["file_a"],
                "pair_b_index": p["idx_b"],
                "pair_b_file": p["file_b"],
                "similarity_score": p["similarity"],
                "hamming_distance": p["hamming_distance"],
                "is_exact_sha256": p["is_exact_sha256"],
                "is_near_duplicate": p["similarity"] >= sim_threshold
            })
        df_pairwise = pd.DataFrame(pairwise_rows)
        csv_pairwise = df_pairwise.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Pairwise Similarity CSV",
            data=csv_pairwise,
            file_name="gan_pairwise_similarity.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 3. Full Markdown Report
    with dl_col3:
        full_report_text = f"""# GAN Image Quality and Diversity Analysis Report

Generated by: GAN Image Quality and Diversity Analyzer
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 1. Dataset Overview
- Source: {source_meta.get('source', 'User Dataset')}
- Total Images: {len(dataset)}
- Random Seed: {source_meta.get('seed', 'N/A')}
- Similarity Threshold: {sim_threshold}

## 2. Diversity Assessment
- Classification: {diversity_rep['classification']}
- Rationale: {diversity_rep['rationale']}
- Total Image Pairs: {diversity_rep['total_pairs']}
- Unique SHA-256 Hashes: {diversity_rep['unique_sha256_count']} ({diversity_rep['uniqueness_ratio_sha']*100:.1f}%)
- Unique dHash Hashes: {diversity_rep['unique_dhash_count']} ({diversity_rep['uniqueness_ratio_dhash']*100:.1f}%)
- Mean Pairwise Similarity: {diversity_rep['mean_similarity']:.4f}
- Median Pairwise Similarity: {diversity_rep['median_similarity']:.4f}
- Max Pairwise Similarity: {diversity_rep['max_similarity']:.4f}
- Min Pairwise Similarity: {diversity_rep['min_similarity']:.4f}
- Flagged Near-Duplicate Pairs: {diversity_rep['near_duplicate_count']} ({diversity_rep['near_duplicate_pct']:.1f}%)

## 3. Image Quality Summary
- Average Brightness: {avg_bright:.2f} / 255
- Average RMS Contrast: {avg_contrast:.2f}
- Average Laplacian Sharpness: {avg_sharpness:.2f}
- Average HSV Saturation: {avg_sat:.2f} / 255

## 4. Methodological Limitations & Transparency Note
- Sharpness Estimate: Uses Discrete Laplacian operator variance. High-frequency sensor noise, checkerboard artifacts, or ringing from transposed convolutions can artificially inflate this score.
- Diversity Heuristic: Based on Difference Hashing (dHash) horizontal gradient comparisons. It assesses intra-batch structural variation and repetition, but does not benchmark against true distribution manifolds (such as FID or Precision/Recall).
"""
        st.download_button(
            label="📥 Full Markdown Report",
            data=full_report_text.encode('utf-8'),
            file_name="gan_diversity_full_report.md",
            mime="text/markdown",
            use_container_width=True
        )
