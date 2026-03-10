#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="cambria_principled"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${ROOT_DIR}/logs/cambria_principled"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_ROOT}/${RUN_STAMP}"
VAST1_ENV_ACTIVATE="${VAST1_ENV_ACTIVATE:-}"

COMMON_ARGS=(
  "example=cambria_shallows_vision_quality"
  "trainer.total_timesteps=150000"
  "trainer.callbacks.eval_callback.eval_freq=25000"
  "trainer.callbacks.eval_callback.n_eval_episodes=3"
  "trainer.callbacks.eval_callback.render=false"
  "env.frame_skip=5"
)

BASELINE_ARGS=(
  "task=cambria_tracking"
  "trainer/callbacks=cambria_multiview"
  "env/renderer=cambria_tracking"
  "trainer.total_timesteps=150000"
  "trainer.callbacks.eval_callback.eval_freq=25000"
  "trainer.callbacks.eval_callback.n_eval_episodes=3"
  "trainer.callbacks.eval_callback.render=false"
  "env.frame_skip=5"
  "env.add_overlays=false"
  "env.debug_overlays_size=0"
)

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found in PATH." >&2
  exit 1
fi

if [[ -n "${VAST1_ENV_ACTIVATE}" ]]; then
  if [[ ! -f "${VAST1_ENV_ACTIVATE}" ]]; then
    echo "VAST1_ENV_ACTIVATE does not exist: ${VAST1_ENV_ACTIVATE}" >&2
    exit 1
  fi
fi

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists." >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

build_cmd() {
  local gpu="$1"
  shift

  local q_root q_activate q_arg cmd
  printf -v q_root '%q' "${ROOT_DIR}"
  cmd="cd ${q_root}"

  if [[ -n "${VAST1_ENV_ACTIVATE}" ]]; then
    printf -v q_activate '%q' "${VAST1_ENV_ACTIVATE}"
    cmd+=" && source ${q_activate}"
  fi

  cmd+=" && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${gpu}"
  cmd+=" && bash scripts/run.sh cambrian/main.py --train"

  local arg
  for arg in "${COMMON_ARGS[@]}"; do
    printf -v q_arg '%q' "${arg}"
    cmd+=" ${q_arg}"
  done
  for arg in "$@"; do
    printf -v q_arg '%q' "${arg}"
    cmd+=" ${q_arg}"
  done

  printf '%s\n' "${cmd}"
}

build_baseline_cmd() {
  local gpu="$1"
  shift

  local q_root q_activate q_arg cmd
  printf -v q_root '%q' "${ROOT_DIR}"
  cmd="cd ${q_root}"

  if [[ -n "${VAST1_ENV_ACTIVATE}" ]]; then
    printf -v q_activate '%q' "${VAST1_ENV_ACTIVATE}"
    cmd+=" && source ${q_activate}"
  fi

  cmd+=" && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${gpu}"
  cmd+=" && bash scripts/run.sh cambrian/main.py --train"

  local arg
  for arg in "${BASELINE_ARGS[@]}"; do
    printf -v q_arg '%q' "${arg}"
    cmd+=" ${q_arg}"
  done
  for arg in "$@"; do
    printf -v q_arg '%q' "${arg}"
    cmd+=" ${q_arg}"
  done

  printf '%s\n' "${cmd}"
}

attach_pane_log() {
  local target="$1"
  local logfile="$2"
  tmux pipe-pane -o -t "${target}" "cat >> ${logfile}"
}

CMD_BASELINE="$(build_baseline_cmd 0 \
  "expname=cambria_baseline_eye" \
  "env/agents/eyes@env.agents.agent.eyes.eye=eye" \
  "env.agents.agent.eyes.eye.resolution=[48,48]" \
  "env.agents.agent.eyes.eye.fov=[72,54]")"

CMD_STATIC_OPTICS="$(build_cmd 0 \
  "expname=cambria_static_optics" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[48,48]" \
  "env.agents.agent.eyes.eye.fov=[72,54]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.55")"

CMD_EVOLVING_OPTICS="$(build_cmd 1 \
  "expname=cambria_evolving_optics" \
  "evo=optics_morphogenesis" \
  "evo.population_size=3" \
  "evo.num_generations=8" \
  "hydra.sweeper.optim.num_workers=3" \
  "trainer.total_timesteps=75000" \
  "trainer.callbacks.eval_callback.eval_freq=15000" \
  "--multirun")"

tmux new-session -d -s "${SESSION_NAME}" -n controls
tmux set-option -t "${SESSION_NAME}" remain-on-exit on
tmux set-option -t "${SESSION_NAME}" mouse on
tmux send-keys -t "${SESSION_NAME}:controls.0" "${CMD_BASELINE}" C-m
attach_pane_log "${SESSION_NAME}:controls.0" "${LOG_DIR}/baseline_eye.log"
tmux split-window -h -t "${SESSION_NAME}:controls"
tmux send-keys -t "${SESSION_NAME}:controls.1" "${CMD_STATIC_OPTICS}" C-m
attach_pane_log "${SESSION_NAME}:controls.1" "${LOG_DIR}/static_optics.log"
tmux select-layout -t "${SESSION_NAME}:controls" tiled

tmux new-window -t "${SESSION_NAME}" -n evolution
tmux send-keys -t "${SESSION_NAME}:evolution.0" "${CMD_EVOLVING_OPTICS}" C-m
attach_pane_log "${SESSION_NAME}:evolution.0" "${LOG_DIR}/evolving_optics.log"

tmux select-window -t "${SESSION_NAME}:controls"

echo "Started tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Pane logs: ${LOG_DIR}"
echo
echo "[Control] baseline eye"
echo "  ${CMD_BASELINE}"
echo "[Control] static optics"
echo "  ${CMD_STATIC_OPTICS}"
echo "[Evolution] evolving optics"
echo "  ${CMD_EVOLVING_OPTICS}"
