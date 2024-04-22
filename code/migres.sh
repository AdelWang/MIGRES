MAX_ITER=5
RETRIEVAL_SET=bm25

RELEVANCE=3.0
MODEL_SET=3
TOPK=5
INPUT_PATH="./data/wikihop.json"
OUTPUT_DIR="./data/migres_res/"
SAVE_PATH=$OUTPUT_DIR/migres_$DATASET\_$MODEL_SET\_$RELEVANCE.json
python pipeline.py \
    --data_path $INPUT_PATH \
    --save_path $SAVE_PATH \
    --top_k $TOPK \
    --k1 0.9 \
    --bm25_b 0.4 \
    --max_iter $MAX_ITER \
    --demon_path "./data/demons/multihopQA_demons.json" \
    --gpt_knowledge \
    --entail_judge \
    --index_dir $INDEX_DIR \
    --normalize \
    --search_type $RETRIEVAL_SET \
    --num_process 64 \
    --num_process_data 200 \
    --model_name $MODEL_SET \
    --num_return 1 \
    --relevance $RELEVANCE