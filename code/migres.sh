#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Set these paths to your dataset and its original/augmented BM25 index.
: "${INPUT_PATH:?Set INPUT_PATH to the benchmark JSON/JSONL file}"
: "${INDEX_DIR:?Set INDEX_DIR to the matching BM25 index}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

RELEVANCE="${RELEVANCE:-3.0}"
SAVE_PATH="${SAVE_PATH:-./results/migres.json}"
API_MODEL="${API_MODEL:-gpt-3.5-turbo-1106}"

python pipeline.py \
    --data_path "$INPUT_PATH" \
    --save_path "$SAVE_PATH" \
    --index_dir "$INDEX_DIR" \
    --api_model "$API_MODEL" \
    --top_k 5 --k1 0.9 --bm25_b 0.4 \
    --max_iter 5 --gpt_knowledge --entail_judge \
    --search_type bm25 --num_process 64 --num_process_data 200 \
    --num_return 1 --relevance "$RELEVANCE" "$@"
