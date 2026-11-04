#!/bin/sh
set -eu
: "${GYM_RESEARCH_FLOPS_LIMIT:?An explicit finite research allocation is required}"
: "${GYM_TRIAL_ID:?A unique trusted trial identity is required}"
python -m nexttransformer_gym.cli --state /state --profiles /opt/gym/profiles.json --platform /opt/platform bootstrap \
  --flops-limit "$GYM_RESEARCH_FLOPS_LIMIT" --trial-id "$GYM_TRIAL_ID" \
  --agent-credential /run/agent-auth/credential.json --verifier-credential /run/verifier-auth/credential.json
exec python -m nexttransformer_gym.cli --state /state --profiles /opt/gym/profiles.json --platform /opt/platform serve --host 0.0.0.0 --container-port 8080
