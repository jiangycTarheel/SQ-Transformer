export DATA_NAME=scan_jump_x2_v2_1000prim
export ARCH=vq_transformer4parsing_3l4h256d512ffn
export LR=5e-4
export DROPOUT=0.3
export NUM_CODES=4
export ENC_NUM_CODES=6

export DATA_BIN=./data/${DATA_NAME}
export USER_DIR=./ar_seq2seq
export SEED=16

export RUN_ID=00-sal
export MAX_TOKENS=4096

export SAVE_DIR=out/${RUN_ID}/${ARCH}-EncC${ENC_NUM_CODES}-DecC${NUM_CODES}/${DATA_NAME}-s${SEED}
export LOG_DIR=${SAVE_DIR}/log


python3 train.py ${DATA_BIN} \
    --enc-dont-update-code-elsewhere \
    --dec-dont-update-code-elsewhere \
    --max-tokens ${MAX_TOKENS} \
    --enc-xtra-pad-code \
    --enc-predict-z-input 'codes' \
    --enc-predict-masked-code \
    --data-name ${DATA_NAME} \
    --enc-lamda 0.999 \
    --lamda 0.999 \
    --latent-use "input_inference&predict_output" \
    --enc-output-vq-l1-factor 0.5 \
    --enc-output-vq-l2-factor 0.1 \
    --dec-output-vq-l1-factor 0.5 \
    --dec-output-vq-l2-factor 0.1 \
    --dec-vq-use-shadow-attn \
    --enc-vq-use-shadow-attn \
    --encoder-attn-k 'encoder_out' \
    --encoder-attn-v 'encoder_out' \
    --weight-decay 0.1 \
    --vq-encoder \
    --enc-latent-use 'input&predict_output' \
    --enc-vq-input 'prev_vq_target_straightthru' \
    --left-pad-source 'False' \
    --vq-enc-l2-factor 0.1 \
    --vq-encoder-xentropy-factor 0.1 \
    --vq-encoder-maximize-z-entropy-factor 0.1 \
    --enc-predict-code \
    --enc-num-codes ${ENC_NUM_CODES} \
    --vq-input "prev_vq_target_straightthru" \
    --no-token-positional-embeddings \
    --vq-l2-factor 0.1 \
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
    --optimizer 'adam' --adam-betas '(0.9, 0.98)' --clip-norm 0 \
    --lr ${LR} --lr-scheduler 'inverse_sqrt' --warmup-updates 4000 \
    --dropout ${DROPOUT} \
    --criterion 'generic_vq_cross_entropy' \
    --seed ${SEED} \
    --eval-overall-acc \
    --eval-acc-args '{"beam": 1}' \
    --best-checkpoint-metric 'overall_acc' \
    --maximize-best-checkpoint-metric \
    --user-dir ${USER_DIR} \
    --tensorboard-logdir ${LOG_DIR}\
    --save-dir ${SAVE_DIR}