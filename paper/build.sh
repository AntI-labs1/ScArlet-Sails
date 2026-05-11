#!/usr/bin/env bash
# Build paper artifacts: PDF, LaTeX (for arXiv), HTML.
#
# Prerequisites:
#   - pandoc (https://pandoc.org/)
#   - LaTeX distribution (TeX Live / MacTeX / MiKTeX with xelatex)
#   - For HTML: pandoc with --to=html5
#
# Usage:
#   ./paper/build.sh        # builds all outputs
#   ./paper/build.sh pdf    # PDF only
#   ./paper/build.sh tex    # LaTeX only (for arXiv submission)
#   ./paper/build.sh html   # HTML only (for Medium/Substack preview)
#   ./paper/build.sh clean  # remove build artifacts
#
# Outputs:
#   paper/build/main.pdf  — submission-ready PDF
#   paper/build/main.tex  — LaTeX source for arXiv
#   paper/build/main.html — HTML preview

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER_DIR="$REPO_ROOT/paper"
DRAFTS_DIR="$PAPER_DIR/drafts"
BUILD_DIR="$PAPER_DIR/build"
SOURCE_MD="$DRAFTS_DIR/main.md"
BIB_FILE="$DRAFTS_DIR/references.bib"

mkdir -p "$BUILD_DIR"

build_pdf() {
    echo "Building PDF..."
    pandoc "$SOURCE_MD" \
        --bibliography="$BIB_FILE" \
        --citeproc \
        --pdf-engine=xelatex \
        --metadata=link-citations:true \
        --metadata=reference-section-title:"References" \
        --variable=geometry:margin=1in \
        --variable=fontsize:11pt \
        --variable=mainfont:"Helvetica" \
        --variable=monofont:"Menlo" \
        --number-sections \
        --toc \
        --toc-depth=2 \
        --resource-path="$PAPER_DIR/figures" \
        -o "$BUILD_DIR/main.pdf"
    echo "  → $BUILD_DIR/main.pdf"
}

build_tex() {
    echo "Building LaTeX (for arXiv)..."
    pandoc "$SOURCE_MD" \
        --bibliography="$BIB_FILE" \
        --natbib \
        --standalone \
        --variable=documentclass:article \
        --variable=geometry:margin=1in \
        --number-sections \
        -o "$BUILD_DIR/main.tex"
    # Copy bib file alongside for arXiv submission
    cp "$BIB_FILE" "$BUILD_DIR/references.bib"
    # Copy figures alongside
    if [ -d "$PAPER_DIR/figures" ]; then
        cp -r "$PAPER_DIR/figures" "$BUILD_DIR/"
    fi
    echo "  → $BUILD_DIR/main.tex (+ references.bib + figures/)"
    echo "  To submit to arXiv, tar these files and upload:"
    echo "    cd $BUILD_DIR && tar -czf arxiv_submission.tar.gz main.tex references.bib figures/"
}

build_html() {
    echo "Building HTML (preview)..."
    pandoc "$SOURCE_MD" \
        --bibliography="$BIB_FILE" \
        --citeproc \
        --standalone \
        --to=html5 \
        --metadata=link-citations:true \
        --metadata=reference-section-title:"References" \
        --number-sections \
        --toc \
        --toc-depth=2 \
        --css=https://cdn.jsdelivr.net/npm/water.css@2/out/water.css \
        --resource-path="$PAPER_DIR/figures" \
        -o "$BUILD_DIR/main.html"
    echo "  → $BUILD_DIR/main.html"
}

build_medium() {
    if [ ! -f "$DRAFTS_DIR/medium.md" ]; then
        echo "Skipping medium build — $DRAFTS_DIR/medium.md not found"
        return
    fi
    echo "Building Medium-adapted HTML..."
    pandoc "$DRAFTS_DIR/medium.md" \
        --standalone \
        --to=html5 \
        --metadata=title:"Why I closed my 8-month algo-trading project (honest negative results)" \
        --css=https://cdn.jsdelivr.net/npm/water.css@2/out/water.css \
        -o "$BUILD_DIR/medium.html"
    echo "  → $BUILD_DIR/medium.html"
    echo "  Copy/paste content into Medium editor; images upload separately."
}

clean() {
    echo "Cleaning $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    echo "  Done."
}

cmd="${1:-all}"
case "$cmd" in
    pdf)    build_pdf ;;
    tex)    build_tex ;;
    html)   build_html ;;
    medium) build_medium ;;
    clean)  clean ;;
    all)
        build_pdf
        build_tex
        build_html
        build_medium
        ;;
    *)
        echo "Usage: $0 [pdf|tex|html|medium|all|clean]"
        exit 1
        ;;
esac

echo ""
echo "Build complete. Artifacts in: $BUILD_DIR"
ls -lh "$BUILD_DIR" 2>/dev/null || true
