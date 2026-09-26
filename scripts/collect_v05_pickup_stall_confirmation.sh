#!/usr/bin/env bash
set -euo pipefail

output_root="artifacts/v05/task4_pickup_stall_confirmation"

for seed_start in 8000 9000 10000 11000 12000 13000; do
  python scripts/policy_record.py \
    --task-id 4 \
    --initial-state-start 0 \
    --seed "${seed_start}" \
    --num-episodes 30 \
    --max-steps 280 \
    --output-root "${output_root}"
done

python scripts/build_stage_cohort_manifest.py \
  docs/manifests/v05_task4_pickup_stall_confirmation_protocol.json \
  "${output_root}" \
  --output docs/manifests/v05_task4_pickup_stall_confirmation_cohort.json
