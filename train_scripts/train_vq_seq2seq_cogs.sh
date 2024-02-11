export DATA_NAME=cogs

export ARCH=vq_transformer4parsing_3l4h256d512ffn
export LR=5e-4
export DROPOUT=0.3
export NUM_CODES=16
export ENC_NUM_CODES=32

export DATA_BIN=./data/${DATA_NAME}
export USER_DIR=./ar_seq2seq
export SEED=10

export RUN_ID=00-srl

export SAVE_DIR=out/${RUN_ID}/${ARCH}-EncC${ENC_NUM_CODES}-DecC${NUM_CODES}/${DATA_NAME}-s${SEED}
export LOG_DIR=${SAVE_DIR}/log

python3 train.py ${DATA_BIN} \
    --enc-dont-update-code-elsewhere \
    --dec-dont-update-code-elsewhere \
    --vq-no-straightthru \
    --enc-lamda 1.0 \
    --lamda 1.0 \
    --enc-vq-encode-proto-x \
    --dec-vq-decode-proto-x \
    --vq-l2-factor 0 \
    --vq-enc-x-proto-x-similarity-l2-factor 0.001 \
    --vq-dec-x-proto-x-similarity-l2-factor 0.001 \
    --enc-output-vq-l1-factor 0.5 \
    --enc-output-vq-l2-factor 0.1 \
    --dec-output-vq-l1-factor 0.1 \
    --dec-output-vq-l2-factor 0.01 \
    --vq-encoder \
    --data-name ${DATA_NAME} \
    --enc-latent-use 'input&predict_proto_output' \
    --enc-vq-input 'prev_vq_target_straightthru' \
    --left-pad-source 'False' \
    --vq-encoder-xentropy-factor 0.1 \
    --vq-encoder-maximize-z-entropy-factor 0.1 \
    --vq-enc-l2-factor 0.1 \
    --enc-predict-code \
    --enc-num-codes ${ENC_NUM_CODES} \
    --latent-use "input_inference&predict_proto_output" \
    --vq-input "prev_vq_target_straightthru" \
    --encoder-attn-k 'proto_encoder_out' \
    --encoder-attn-v 'proto_encoder_out' \
    --save-interval-updates 1000 \
    --no-token-positional-embeddings \
    --vq-maximize-z-entropy-factor 0.1 \
    --vq-xentropy-factor 0.1 \
    --latent-factor 0 \
    --num-codes ${NUM_CODES} \
    --task seq2seq-translation \
    --block-cls highway \
    --self-attn-cls shaw \
    --enc-self-attn-cls shaw \
    --enc-block-cls highway \
    --max-rel-positions 20 \
    --save-interval-updates 1000 \
    --max-update 1000000 \
    --keep-best-checkpoints 3 \
    --keep-interval-updates 3 \
    --no-epoch-checkpoints \
    --arch ${ARCH} --share-decoder-input-output-embed \
    --optimizer 'adam' --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr ${LR} --lr-scheduler 'inverse_sqrt' --warmup-updates 4000 \
    --dropout ${DROPOUT} \
    --criterion 'generic_vq_cross_entropy' \
    --max-tokens 4096 \
    --seed ${SEED} \
    --eval-overall-acc \
    --eval-acc-args '{"beam": 1}' \
    --best-checkpoint-metric 'overall_acc' \
    --maximize-best-checkpoint-metric \
    --user-dir ${USER_DIR} \
    --tensorboard-logdir ${LOG_DIR}\
    --save-dir ${SAVE_DIR}