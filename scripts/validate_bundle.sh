#!/usr/bin/env bash
# Framework-native checks of a bundle split: static check, then materialize one sample per stage (CPU only).
# Usage: BUNDLE=<bundle>/train scripts/validate_bundle.sh
set -uo pipefail
: "${COSMOS_FRAMEWORK_DIR:?}" "${BUNDLE:?}"
cd "$COSMOS_FRAMEWORK_DIR"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 CUDA_VISIBLE_DEVICES=""
python -m cosmos_framework.scripts.validate_bimanual_data --bundle "$BUNDLE" > /tmp/validate_static.json 2>&1 \
  && echo "static: PASSED" || { echo "static: FAILED (see /tmp/validate_static.json)"; }
for st in 0 1 2 3; do
  python -m cosmos_framework.scripts.validate_bimanual_data --bundle "$BUNDLE" --materialize --stage $st \
    > /tmp/validate_stage$st.log 2>&1 && grep -q '"materialized": true' /tmp/validate_stage$st.log \
    && echo "stage $st materialize: PASSED" || echo "stage $st materialize: FAILED (see /tmp/validate_stage$st.log)"
done
