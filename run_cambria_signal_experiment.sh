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
export PYTHONUNBUFFERED=1

cd "${ROOT_DIR}"

python cambrian/main.py --train \
  example=cambria_detection_vision_quality \
  expname=cambria_detection_signal \
  evo=optics_morphogenesis \
  evo.population_size=4 \
  evo.num_generations=10 \
  hydra.sweeper.optim.num_workers=4 \
  trainer.total_timesteps=50000 \
  trainer.callbacks.eval_callback.eval_freq=10000 \
  trainer.callbacks.eval_callback.n_eval_episodes=3 \
  trainer.callbacks.eval_callback.render=false \
  --multirun
