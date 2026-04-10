cd /home/wyu3/workspace/opd

# python3 analysis/prepare_rollout_dataset.py \
#   --input "/project/flame/wyu3/mopd/rollout/summary_from_all-SuccessFalseFailFalse-chemistry-SDPO-train32-alpha0.5-rollout8-lr1e-5-lambda0.0-clip_adv_highnull-drossTrue-Qwen-Qwen3-4B-Instruct-2507-local_sdpo" \
#   --output-dir analysis_outputs/chemistry_rollout_dataset \
#   --max-prompts 300 \
#   --max-responses-per-prompt 8

python3 analysis/build_context_variants.py \
  --candidates analysis_outputs/chemistry_rollout_dataset/candidate_responses.jsonl \
  --evidence analysis_outputs/chemistry_rollout_dataset/evidence_items.jsonl \
  --output analysis_outputs/chemistry_context_variants.jsonl

# python3 analysis/compute_teacher_signal_metrics.py \
#   --input analysis_outputs/teacher_scores.jsonl \
#   --output analysis_outputs/teacher_metrics.json
