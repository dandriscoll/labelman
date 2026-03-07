#!/usr/bin/env bash
# Built-in BLIP integration script for labelman.
# Sends images to a BLIP HTTP endpoint and returns captions as JSON lines.
#
# Usage: blip.sh --endpoint URL [--prompt TEXT] --images FILE [FILE ...]
#
# Output (stdout): one JSON line per image:
#   {"image": "path/to/img.jpg", "caption": "a photo of a cat"}
#
# The endpoint should accept POST with multipart/form-data:
#   - file: the image file
#   - prompt: (optional) question or prompt text
# and return JSON: {"caption": "..."}

set -euo pipefail

ENDPOINT=""
PROMPT=""
IMAGES=()
parsing_images=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
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

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "Error: --images requires at least one file" >&2
    exit 1
fi

for img in "${IMAGES[@]}"; do
    if [[ ! -f "$img" ]]; then
        echo "Error: file not found: $img" >&2
        exit 1
    fi

    curl_args=(-s -X POST "$ENDPOINT" -F "file=@$img")
    if [[ -n "$PROMPT" ]]; then
        curl_args+=(-F "prompt=$PROMPT")
    fi

    response=$(curl "${curl_args[@]}")
    caption=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['caption'])")
    python3 -c "import json; print(json.dumps({'image': '$img', 'caption': '''$caption'''}))"
done
