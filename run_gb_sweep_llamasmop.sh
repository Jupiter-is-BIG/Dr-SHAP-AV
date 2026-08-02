#!/usr/bin/env bash
# Gaussian-blur (GB) visual-severity sweep for Llama-SMoP, full LRS3 test set,
# Permutation SHAP. Launches all 6 configs (clean + levels 1-5) in parallel,
# one per GPU, throttled to the GPU pool size below.
set -uo pipefail

# ---- EDIT ME ---------------------------------------------------------------
GPUS=(0 1 2 3 4 5)   # GPU ids this script may use (6 configs -> 6 GPUs fits in one shot)
ROOT=CHANGE_ME        # /root/directory/path (contains labels/ + preprocessed LRS3)
CKPT=CHANGE_ME         # path to LRS3_SMoP_audiovisual_..._pool-4-2_seed7
AVH_CKPT=CHANGE_ME     # path to AV-HuBERT Large ckpt (large_vox_iter5.pt)
OUT_DIR=CHANGE_ME      # dir to save per-sample SHAP .npz files (must exist already)
WANDB_PROJECT=dr-shap-av-visual
TEST_FILE=lrs3_test_transcript_lengths_seg24s_LLM_lowercase.csv
# -----------------------------------------------------------------------------

for v in ROOT CKPT AVH_CKPT OUT_DIR; do
  if [ "${!v}" = "CHANGE_ME" ]; then
    echo "ERROR: please edit $v at the top of this script before running." >&2
    exit 1
  fi
done

LOGDIR=logs/gb_sweep_llamasmop
mkdir -p "$LOGDIR"

COMMON_ARGS=(--wandb-project "$WANDB_PROJECT" --root-dir "$ROOT" --pretrained-model-path "$CKPT" \
  --modality audiovisual --pretrain-avhubert-enc-video-path "$AVH_CKPT" --audio-encoder-name openai/whisper-medium.en \
  --rank 32 --alpha 4 --llm-model meta-llama/Llama-3.2-1B --unfrozen-modules peft_llm --add-PEFT-LLM lora \
  --downsample-ratio-audio 4 --downsample-ratio-video 2 --test-file "$TEST_FILE" \
  --MoP-experts 4 --MoP-topk 2 \
  --compute-shap True --shap-alg permutation --num-samples-shap 2000 --output-path-shap "$OUT_DIR" --seed 42)

# suffix:extra-args (empty extra-args = clean baseline)
JOBS=(
  "clean:"
  "lvl1:--vid-dist-type GB --vid-dist-level 1"
  "lvl2:--vid-dist-type GB --vid-dist-level 2"
  "lvl3:--vid-dist-type GB --vid-dist-level 3"
  "lvl4:--vid-dist-type GB --vid-dist-level 4"
  "lvl5:--vid-dist-type GB --vid-dist-level 5"
)

MAX_PARALLEL=${#GPUS[@]}
gpu_idx=0

for job in "${JOBS[@]}"; do
  suffix="${job%%:*}"
  extra="${job#*:}"
  gpu="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
  gpu_idx=$((gpu_idx + 1))

  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n
  done

  exp_name="LRS3_LlamaSMoP_shap_permutation_viddist-GB-${suffix}"
  echo "Launching $exp_name on GPU $gpu (log: $LOGDIR/${exp_name}.log)"
  CUDA_VISIBLE_DEVICES=$gpu python eval_LlamaSMoP.py "${COMMON_ARGS[@]}" $extra \
    --exp-name "$exp_name" > "$LOGDIR/${exp_name}.log" 2>&1 &
done

wait
echo "All Llama-SMoP GB severity-sweep runs finished. Logs in $LOGDIR/"
