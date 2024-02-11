# DATA: wmt17_en_de
export DATA_NAME=scan_jump_x2_v2_1000prim
export DATA_BIN=./data/${DATA_NAME}
export USER_DIR=./ar_seq2seq
export RUN_ID=00-sal
export ARCH=vq_transformer4parsing_3l4h256d512ffn-EncC6-DecC4
export SAVE_DIR=out/${RUN_ID}/${ARCH}/${DATA_NAME}-s16
export RESULT_DIR=${SAVE_DIR}/results

python3 test.py ${DATA_BIN} \
    --data-name ${DATA_NAME} \
    --results-path ${RESULT_DIR} \
    --path ${SAVE_DIR}/checkpoint_best.pt \
    --gen-subset test \
    --user-dir ${USER_DIR} \
    --task seq2seq-translation \
    --beam 1 \
    --batch-size 512 \
    --eval-acc-args '{"beam": 1, "max_len_b": 200}'
