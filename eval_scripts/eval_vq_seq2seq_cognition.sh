export DATA_NAME=cognition_cg
export EVAL_DATA_NAME=cognition_cg
export DATA_BIN=./data/${EVAL_DATA_NAME}
export USER_DIR=./ar_seq2seq
export RUN_ID=00-srl
export ARCH=vq_transformer4parsing_6l8h512d1024ffn_6lvq-EncC64-DecC32
export SAVE_DIR=out/${RUN_ID}/${ARCH}/${DATA_NAME}-s16
export RESULT_DIR=${SAVE_DIR}/results

python3 test.py ${DATA_BIN} \
    --data-name ${DATA_NAME} \
    --results-path ${RESULT_DIR} \
    --path ${SAVE_DIR}/checkpoint_best.pt \
    --gen-subset test \
    --user-dir ${USER_DIR} \
    --task seq2seq-translation \
    --beam 5 \
    --batch-size 512 \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-cognition \
    --dataset-impl 'raw' \
    --scoring 'sacrebleu' \
    --remove-bpe
