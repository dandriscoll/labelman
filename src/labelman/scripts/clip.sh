#!/usr/bin/env bash
# Built-in CLIP integration script for labelman.
# Sends images and candidate terms to a CLIP HTTP endpoint.
# Returns per-image, per-term confidence scores as JSON lines.
#
# Usage: clip.sh --endpoint URL --terms TERM1,TERM2,... --images FILE [FILE ...]
#
# Output (stdout): one JSON line per image:
#   {"image": "path/to/img.jpg", "scores": {"term1": 0.85, "term2": 0.12}}
#
# The endpoint should accept POST with:
#   - file: the image file (multipart/form-data)
#   - terms: comma-separated candidate terms (form field)
# and return JSON: {"scores": {"term1": float, ...}}

set -euo pipefail

ENDPOINT=""
TERMS=""
IMAGES=()
parsing_images=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --terms)
            TERMS="$2"
            shift 2
            ;;
        --images)
            shift
            parsing_images=true
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                IMAGES+=("$1")
                shift
            done
            ;;
        *)
            if [[ "$parsing_images" == true ]]; then
                IMAGES+=("$1")
                shift
            else
                echo "Unknown argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$ENDPOINT" ]]; then
    echo "Error: --endpoint is required" >&2
    exit 1
fi

if [[ -z "$TERMS" ]]; then
    echo "Error: --terms is required" >&2
    exit 1
fi

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "Error: --images requires at least one file" >&2
    exit 1
fi

for img in "${IMAGES[@]}"; do
    if [[ ! -f "$img" ]]; then
        echo "Error: file not found: $img" >&2
        exit 1
    fi

    response=$(curl -s -X POST "$ENDPOINT" -F "file=@$img" -F "terms=$TERMS")
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Error: curl failed (code $rc) on: $img" >&2
        exit $rc
    fi
    scores=$(echo "$response" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['scores']))")
    python3 -c "import json; print(json.dumps({'image': $(python3 -c "import json; print(json.dumps('$img'))"), 'scores': $scores}))"
done
