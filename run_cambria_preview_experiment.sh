#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAST1_ENV_ACTIVATE="${VAST1_ENV_ACTIVATE:-${ROOT_DIR}/.venv/bin/activate}"

if [[ ! -f "${VAST1_ENV_ACTIVATE}" ]]; then
  echo "Activate script not found: ${VAST1_ENV_ACTIVATE}" >&2
  exit 1
fi

source "${VAST1_ENV_ACTIVATE}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

cd "${ROOT_DIR}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

PREFLIGHT_EXPNAME="${CAMBRIA_PREVIEW_PREFLIGHT_EXPNAME:-cambria_detection_preview_preflight}"
TRAIN_EXPNAME="${CAMBRIA_PREVIEW_EXPNAME:-cambria_detection_preview}"
HYDRA_LAUNCHER="${CAMBRIA_PREVIEW_HYDRA_LAUNCHER:-basic}"

# Save a pre-training scene/vision/optics snapshot before any RL updates.
python tools/optics/cambria_optics_diagnostic.py \
  example=cambria_detection_preview \
  expname="${PREFLIGHT_EXPNAME}"

# Run a short preview evolution so that evaluation renders appear quickly.
python cambrian/main.py --train \
  example=cambria_detection_preview \
  expname="${TRAIN_EXPNAME}" \
  evo=optics_morphogenesis \
  evo.population_size=2 \
  evo.num_generations=2 \
  hydra.sweeper.optim.num_workers=2 \
  hydra/launcher="${HYDRA_LAUNCHER}" \
  trainer.callbacks.eval_callback.eval_freq=1000 \
  trainer.callbacks.eval_callback.n_eval_episodes=1 \
  trainer.callbacks.eval_callback.render=false \
  --multirun
