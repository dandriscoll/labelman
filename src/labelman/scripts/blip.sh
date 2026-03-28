#!/usr/bin/env bash
# Built-in BLIP integration script for labelman.
# Sends images to a BLIP HTTP endpoint and returns captions as JSON lines.
#
# Usage: blip.sh --endpoint URL [--prompt TEXT] [--max-tokens N] --images FILE [FILE ...]
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
MAX_TOKENS=""
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
        --max-tokens)
            MAX_TOKENS="$2"
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
        # BLIP-2 VQA works best with "Question: ... Answer:" format
        formatted_prompt="Question: $PROMPT Answer:"
        curl_args+=(-F "prompt=$formatted_prompt")
    fi
    if [[ -n "$MAX_TOKENS" ]]; then
        curl_args+=(-F "max_tokens=$MAX_TOKENS")
    fi

    response=$(curl "${curl_args[@]}")
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Error: curl failed (code $rc) on: $img" >&2
        exit $rc
    fi
    echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
text = data.get('caption') or data.get('answer', '')
image = sys.argv[1]
key = 'answer' if 'answer' in data else 'caption'
print(json.dumps({'image': image, key: text}))
" "$img"
done
