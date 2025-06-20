"""streamlit_app.py
Interactive Streamlit front-end for the advanced Word Cloud generator.
Run with:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from wordcloud import STOPWORDS, WordCloud

# Re-use preprocessing + helpers from CLI module
from wordcloud_cli import (
    clean_and_tokenise,
    compute_frequencies,
    lemmatise,
    make_colour_func,
)

st.set_page_config(
    page_title="Sophisticated Word Cloud Generator",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Sidebar – input & customisation controls
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Customisation Panel")

uploaded_text_file = st.sidebar.file_uploader(
    "Upload a text file (\*.txt)", type=["txt"], key="upload_txt"
)
text_input = st.sidebar.text_area(
    "Or paste text below", height=200, placeholder="Enter or paste text here…"
)

mask_file = st.sidebar.file_uploader(
    "Optional: shape mask (PNG/JPG)", type=["png", "jpg", "jpeg"], key="mask"
)

font_file = st.sidebar.file_uploader(
    "Optional: custom font (TTF/OTF)", type=["ttf", "otf"], key="font"
)

st.sidebar.markdown("---")

# Appearance
bg_color = st.sidebar.color_picker("Background colour", value="#ffffff")
contour_color = st.sidebar.color_picker("Contour colour", value="#000000")
contour_width = st.sidebar.slider("Contour width", 0, 10, 0)

max_words = st.sidebar.slider("Max words", 50, 1000, 200, step=50)
min_font = st.sidebar.slider("Min. font size", 4, 50, 10)
max_font = st.sidebar.slider("Max. font size", 50, 300, 200)

width_px = st.sidebar.slider("Image width (px)", 400, 3000, 1600, step=100)
height_px = st.sidebar.slider("Image height (px)", 200, 2000, 800, step=100)
scale = st.sidebar.slider("Scale (resolution multiplier)", 1, 5, 1)
prefer_horizontal = st.sidebar.slider(
    "% words horizontal", 0, 100, 90, help="Higher → more horizontal words"
)
collocations = st.sidebar.checkbox(
    "Show collocations (multi-word phrases)", value=False, help="Turn on to include bi-grams"
)

# Colour strategy
colour_mode = st.sidebar.radio(
    "Colour strategy", ("Matplotlib colormap", "Custom palette")
)
colormap = None
palette: List[str] | None = None
if colour_mode == "Matplotlib colormap":
    import matplotlib

    # Matplotlib ≥3.7 no longer exposes `cmap_d`; use the new API with fallback
    try:
        cmap_names = sorted(matplotlib.pyplot.colormaps())  # type: ignore[attr-defined]
    except AttributeError:
        # Older Matplotlib versions
        cmap_names = sorted(matplotlib.cm.cmap_d.keys())
    colormap = st.sidebar.selectbox("Choose colormap", cmap_names, index=cmap_names.index("viridis"))
else:
    st.sidebar.markdown("Pick up to 5 colours for palette")
    palette = []
    for i in range(5):
        col = st.sidebar.color_picker(f"Colour {i+1}", value="#" + "000000"[i:] + "fff"[i:])
        if col:
            palette.append(col)
    if not palette:
        st.sidebar.warning("At least one colour must be chosen – defaulting to black.")
        palette = ["#000000"]

st.sidebar.markdown("---")
remove_sw = st.sidebar.checkbox("Remove English stop-words", value=True)
extra_stopwords = st.sidebar.text_input("Extra stop-words (comma-separated)")

st.sidebar.markdown("---")
if st.sidebar.button("Generate Word Cloud", type="primary"):
    st.session_state["generate"] = True
else:
    st.session_state.setdefault("generate", False)

# -----------------------------------------------------------------------------
# Main UI – header & output panels
# -----------------------------------------------------------------------------
st.title("✨ Sophisticated Word Cloud Generator")

with st.expander("About this app", expanded=False):
    st.markdown(
        """
        Create beautifully customised word-clouds with advanced NLP pre-processing, high-resolution exports,
        and a modern ⚡ interactive interface. Built with **Streamlit** + **WordCloud** + **NLTK**.
        """
    )

# Optional Lottie animation header (non-blocking)
try:
    import streamlit_lottie as st_lottie
    import json, requests

    def load_lottie_url(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_json = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_cf6yoy.json")
    if lottie_json:
        st_lottie.st_lottie(lottie_json, height=150, key="header_anim")
except Exception:
    pass  # Lottie optional – ignore failures

# Early exit until user clicks generate
if not st.session_state["generate"]:
    st.info("👈 Configure options then click **Generate Word Cloud**")
    st.stop()

# -----------------------------------------------------------------------------
# Gather text
# -----------------------------------------------------------------------------
if uploaded_text_file is not None:
    text = uploaded_text_file.read().decode("utf-8", errors="ignore")
elif text_input.strip():
    text = text_input
else:
    st.error("Please provide some input text (upload or paste).")
    st.stop()

# -----------------------------------------------------------------------------
# Pre-processing
# -----------------------------------------------------------------------------
with st.spinner("Analysing text …"):
    tokens = clean_and_tokenise(text, remove_stopwords=remove_sw)

    if extra_stopwords.strip():
        extras = {w.strip().lower() for w in extra_stopwords.split(",") if w.strip()}
        tokens = [t for t in tokens if t not in extras]

    tokens = lemmatise(tokens)
    freqs = compute_frequencies(tokens)

    if not freqs:
        st.error("No valid words found after processing. Try different settings.")
        st.stop()

# -----------------------------------------------------------------------------
# Prepare mask & font
# -----------------------------------------------------------------------------
mask_array = None
if mask_file is not None:
    try:
        mask_array = np.array(Image.open(mask_file))
    except Exception as e:
        st.warning(f"Could not load mask image: {e}")

font_path = None
if font_file is not None:
    font_temp = Path("_uploaded_font.ttf")
    font_temp.write_bytes(font_file.read())
    font_path = str(font_temp)

# -----------------------------------------------------------------------------
# Build word cloud
# -----------------------------------------------------------------------------
wc_kwargs = dict(
    background_color=bg_color,
    contour_color=contour_color,
    contour_width=contour_width,
    max_words=max_words,
    min_font_size=min_font,
    max_font_size=max_font,
    width=width_px,
    height=height_px,
    prefer_horizontal=prefer_horizontal / 100,
    collocations=collocations,
    scale=scale,
    font_path=font_path,
    stopwords=STOPWORDS,
    mask=mask_array,
)

if palette:
    wc_kwargs["color_func"] = make_colour_func(palette)
else:
    wc_kwargs["colormap"] = colormap

wc = WordCloud(**wc_kwargs).generate_from_frequencies(freqs)

# -----------------------------------------------------------------------------
# Display results
# -----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Generated Word Cloud")
    image_bytes = io.BytesIO()
    wc.to_image().save(image_bytes, format="PNG")
    st.image(image_bytes.getvalue(), use_container_width=True)

    st.download_button(
        label="💾 Download Word Cloud as PNG",
        data=image_bytes.getvalue(),
        file_name="wordcloud.png",
        mime="image/png",
    )

with col2:
    st.subheader("Top Word Frequencies")
    top_n = st.slider("Show top N words", 5, 50, 20, key="topn_slider")
    top_items = freqs.most_common(top_n)
    # Adjust figure height proportionally so whitespace decreases with fewer bars
    fig_height = max(2, 0.3 * top_n + 1)  # 0.3" per bar + 1" padding
    freq_fig, ax = plt.subplots(figsize=(4, fig_height))
    words, counts = zip(*top_items)
    ax.barh(words[::-1], counts[::-1], color="steelblue")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    st.pyplot(freq_fig, use_container_width=True)

    # Download frequency CSV
    csv_buf = io.StringIO()
    csv_buf.write("word,frequency\n")
    csv_buf.writelines(f"{w},{c}\n" for w, c in freqs.items())
    st.download_button(
        "📊 Download full frequencies (CSV)",
        data=csv_buf.getvalue(),
        file_name="frequencies.csv",
        mime="text/csv",
    )

st.success("Done! Enjoy your customised word cloud ✨") 