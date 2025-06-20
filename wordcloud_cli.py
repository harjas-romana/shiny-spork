#!/usr/bin/env python3
"""wordcloud_cli.py
A fully-featured command-line interface for generating highly-customisable
and informative word-clouds.

Features
--------
1. Extensive appearance customisation (background/contour, fonts, colours, shape masks).
2. NLP-based preprocessing with stop-word removal and lemmatisation.
3. Word-frequency statistics with optional bar-chart visualisation.
4. High-resolution export to PNG/JPG/SVG (timestamped filenames).
5. CSV/JSON export of frequency data.
6. Interactive preview before saving (can be bypassed with --auto-save).

Usage examples
--------------
# Basic generation from a text file with default settings
python wordcloud_cli.py --input-file speech.txt

# Custom colours, mask, and export formats
python wordcloud_cli.py \
    --input-file reviews.txt \
    --mask-image logo.png \
    --background white --contour-colour steelblue --contour-width 2 \
    --colormap plasma --max-words 300 --export-formats png,svg

Run ``python wordcloud_cli.py --help`` for the full list of options.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import STOPWORDS, WordCloud

# nltk imports (download resources on first run)
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# NLTK setup – download required corpora on demand
# ---------------------------------------------------------------------------
for pkg in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"):
    try:
        nltk.data.find(pkg)
    except LookupError:  # pragma: no cover – only on first run
        nltk.download(pkg, quiet=True)

# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------
WORDNET_TAG_MAP = {
    "J": wordnet.ADJ,
    "V": wordnet.VERB,
    "N": wordnet.NOUN,
    "R": wordnet.ADV,
}

def _get_wordnet_pos(treebank_tag: str):
    """Map NLTK POS tags to WordNet tags (default to noun)."""
    return WORDNET_TAG_MAP.get(treebank_tag[0], wordnet.NOUN)


def clean_and_tokenise(text: str, remove_stopwords: bool = True) -> List[str]:
    """Lower-case, remove punctuation/non-alphabetic chars, tokenise, and optionally remove stopwords."""
    # Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = word_tokenize(text)

    if remove_stopwords:
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]

    return tokens


def lemmatise(tokens: Iterable[str]) -> List[str]:
    """Lemmatise tokens using part-of-speech information."""
    pos_tags = pos_tag(tokens)
    lemmatiser = WordNetLemmatizer()
    return [lemmatiser.lemmatize(word, _get_wordnet_pos(pos)) for word, pos in pos_tags]


def compute_frequencies(tokens: Iterable[str]) -> Counter:
    """Compute word-frequency counter from an iterable of tokens."""
    return Counter(tokens)

# ---------------------------------------------------------------------------
# WordCloud generation helpers
# ---------------------------------------------------------------------------

def make_colour_func(palette: List[str]):
    """Return a colour-function that randomly picks colours from *palette*."""

    def _colour(word, font_size, position, orientation, random_state=None, **kwargs):  # noqa: D401,E501
        return random.choice(palette)

    return _colour


def generate_wc(
    frequencies: Counter,
    mask_image: Path | None,
    args: argparse.Namespace,
):
    """Generate a WordCloud object using *frequencies* and CLI *args*."""
    mask = None
    if mask_image:
        mask = np.array(Image.open(mask_image))

    wc_kwargs = dict(
        background_color=args.background,
        contour_color=args.contour_colour,
        contour_width=args.contour_width,
        max_words=args.max_words,
        min_font_size=args.min_font_size,
        max_font_size=args.max_font_size,
        font_path=str(args.font_path) if args.font_path else None,
        width=args.width,
        height=args.height,
        stopwords=STOPWORDS,
        mask=mask,
    )

    if args.palette:
        palette = [c.strip() for c in args.palette.split(",") if c.strip()]
        wc_kwargs["color_func"] = make_colour_func(palette)
    else:
        wc_kwargs["colormap"] = args.colormap

    return WordCloud(**wc_kwargs).generate_from_frequencies(frequencies)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_wordcloud(wc: WordCloud, title: str = "Word Cloud"):
    plt.figure(figsize=(wc.width / 100, wc.height / 100))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)


def plot_frequencies(frequencies: Counter, top_n: int = 20):
    words, freqs = zip(*frequencies.most_common(top_n))
    plt.figure(figsize=(10, 6))
    plt.bar(words, freqs, color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(f"Top {top_n} Word Frequencies")
    plt.tight_layout()

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_frequencies_csv(frequencies: Counter, path: Path):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        writer.writerows(frequencies.items())


def export_frequencies_json(frequencies: Counter, path: Path):
    with path.open("w") as f:
        json.dump(dict(frequencies), f, indent=2)

# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return ivalue


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a highly-customisable word cloud from input text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    group_in = p.add_mutually_exclusive_group(required=True)
    group_in.add_argument("--input-file", type=Path, help="Path to text file containing input.")
    group_in.add_argument("--text", type=str, help="Raw text input provided via CLI.")

    p.add_argument("--mask-image", type=Path, help="Optional image mask for custom cloud shape.")
    p.add_argument("--output-prefix", default="wordcloud", help="Prefix for output files (timestamp appended).")
    p.add_argument(
        "--export-formats",
        default="png",
        help="Comma-separated list of image formats to save (png,jpg,svg).",
    )
    p.add_argument("--export-freq", choices=["csv", "json", "both", "none"], default="csv",
                   help="Export word-frequency data format.")

    # Appearance
    p.add_argument("--background", default="white", help="Background colour.")
    p.add_argument("--contour-colour", default="black", help="Contour / outline colour.")
    p.add_argument("--contour-width", type=int, default=0, help="Contour line width.")
    p.add_argument("--palette", help="Comma-separated list of hex/colour names to use as a discrete palette.")
    p.add_argument("--colormap", default="viridis", help="Matplotlib colormap if no palette specified.")
    p.add_argument("--font-path", type=Path, help="Path to .ttf/.otf font file.")
    p.add_argument("--max-words", type=_positive_int, default=200, help="Maximum number of words to display.")
    p.add_argument("--min-font-size", type=_positive_int, default=10, help="Minimum font size.")
    p.add_argument("--max-font-size", type=_positive_int, default=200, help="Maximum font size.")
    p.add_argument("--width", type=_positive_int, default=1600, help="Image width in pixels.")
    p.add_argument("--height", type=_positive_int, default=800, help="Image height in pixels.")

    # Pre-processing
    p.add_argument("--keep-stopwords", action="store_true", help="Do NOT remove English stopwords.")

    # Misc
    p.add_argument("--top-n", type=_positive_int, default=20, help="Number of top words to show in frequency bar chart.")
    p.add_argument("--preview", action="store_true", help="Preview wordcloud and bar chart before saving.")
    p.add_argument("--auto-save", action="store_true", help="Skip confirmation prompt and save immediately.")
    return p


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Read input text
    if args.input_file:
        text = args.input_file.read_text(encoding="utf-8", errors="ignore")
    else:
        text = args.text

    # Preprocess
    tokens = clean_and_tokenise(text, remove_stopwords=not args.keep_stopwords)
    tokens = lemmatise(tokens)
    freqs = compute_frequencies(tokens)

    # Generate cloud
    wc = generate_wc(freqs, args.mask_image, args)

    # Visualise (preview)
    if args.preview:
        plot_wordcloud(wc)
        plot_frequencies(freqs, top_n=args.top_n)
        plt.show()

    # Confirmation prompt unless auto-save
    if not args.auto_save:
        resp = input("Save images/data? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted – nothing saved.")
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.output_prefix}_{timestamp}"

    # Export images
    for fmt in [f.strip().lower() for f in args.export_formats.split(",") if f.strip()]:
        out_path = Path(f"{base}.{fmt}")
        wc.to_file(str(out_path))
        print(f"Saved wordcloud image → {out_path.relative_to(Path.cwd())}")

    # Export frequencies image (bar chart)
    plot_frequencies(freqs, top_n=args.top_n)
    freq_img_path = Path(f"{base}_frequencies.png")
    plt.savefig(freq_img_path, dpi=300)
    print(f"Saved frequency bar chart → {freq_img_path.relative_to(Path.cwd())}")

    # Export frequency data
    if args.export_freq in ("csv", "both"):
        csv_path = Path(f"{base}_frequencies.csv")
        export_frequencies_csv(freqs, csv_path)
        print(f"Saved frequency data (CSV) → {csv_path.relative_to(Path.cwd())}")
    if args.export_freq in ("json", "both"):
        json_path = Path(f"{base}_frequencies.json")
        export_frequencies_json(freqs, json_path)
        print(f"Saved frequency data (JSON) → {json_path.relative_to(Path.cwd())}")

    print("Done!")


if __name__ == "__main__":
    main() 