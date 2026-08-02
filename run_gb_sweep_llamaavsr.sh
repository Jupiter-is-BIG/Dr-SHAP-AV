#!/usr/bin/env bash
# Gaussian-blur (GB) visual-severity sweep for Llama-AVSR, full LRS3 test set,
# Permutation SHAP. Launches all 6 configs (clean + levels 1-5) in parallel,
# one per GPU, throttled to the GPU pool size below.
set -uo pipefail

GPUS=(0 1 2 3 4 5)
ROOT=/ucappell/datasets 
CKPT=/aa4825/models/llama_avsr/LRS3_audiovisual_avg-pooling_AVH-Large_Whisper-M_Llama3.2-1B_pool-4-2_LN_seed7.pth
AVH_CKPT=/aa4825/models/av_hubert/large_vox_iter5.pt
OUT_DIR=output
WANDB_PROJECT=dr-shap-av-visual
TEST_FILE=lrs3_test_transcript_lengths_seg24s_LLM_lowercase.csv

LOGDIR=logs/gb_sweep_llamaavsr
mkdir -p "$LOGDIR"

COMMON_ARGS=(--wandb-project "$WANDB_PROJECT" --root-dir "$ROOT" --pretrained-model-path "$CKPT" \
  --modality audiovisual --pretrain-avhubert-enc-video-path "$AVH_CKPT" --audio-encoder-name openai/whisper-medium.en \
  --rank 32 --alpha 4 --llm-model meta-llama/Llama-3.2-1B --unfrozen-modules peft_llm --add-PEFT-LLM lora \
  --downsample-ratio-audio 4 --downsample-ratio-video 2 --test-file "$TEST_FILE" \
  --compute-shap True --shap-alg permutation --num-samples-shap 2000 --output-path-shap "$OUT_DIR" --seed 42)

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
  exp_name="LRS3_LlamaAVSR_shap_permutation_viddist-GB-${suffix}"
  echo "Launching $exp_name on GPU $gpu (log: $LOGDIR/${exp_name}.log)"
  CUDA_VISIBLE_DEVICES=$gpu python eval_LlamaAVSR.py "${COMMON_ARGS[@]}" $extra \
    --exp-name "$exp_name" > "$LOGDIR/${exp_name}.log" 2>&1 &
done

wait
echo "All Llama-AVSR GB severity-sweep runs finished. Logs in $LOGDIR/"
