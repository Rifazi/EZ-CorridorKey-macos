#!/usr/bin/env bash
# Quick wrapper to remove background from videos using rembg
# Much faster than GVM AUTO on Mac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Run 1-install.sh first!"
    exit 1
fi

source .venv/bin/activate

if [ $# -lt 2 ]; then
    echo "Usage: ./remove_bg.sh <input.mov> <output.mov>"
    echo ""
    echo "Example:"
    echo "  ./remove_bg.sh greenscreen.mov greenscreen_alpha.mov"
    echo ""
    echo "This uses rembg (AI background removal) which is faster than GVM AUTO."
    echo "The output video will have a transparent background."
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
    echo "[ERROR] Input file not found: $INPUT"
    exit 1
fi

echo "Removing background..."
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo ""

python remove_background.py "$INPUT" "$OUTPUT"
