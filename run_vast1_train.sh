#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="evo_morphology"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${ROOT_DIR}/logs/vast1_runs"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_ROOT}/${RUN_STAMP}"
VAST1_ENV_ACTIVATE="${VAST1_ENV_ACTIVATE:-}"

COMMON_ARGS=(
  "--train"
  "task=tracking"
  "env/renderer=renderer"
  "trainer.callbacks.eval_callback.eval_freq=5000"
  "trainer.callbacks.eval_callback.render=false"
  "env.add_overlays=false"
  "env.debug_overlays_size=0"
)

EXTRA_ARGS=("$@")

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found in PATH." >&2
  exit 1
fi

if [[ -n "${VAST1_ENV_ACTIVATE}" ]]; then
  if [[ "${VAST1_ENV_ACTIVATE}" == "/path/to/venv/bin/activate" ]]; then
    echo "VAST1_ENV_ACTIVATE is still set to the placeholder path." >&2
    echo "Unset it, or set it to a real activate script path." >&2
    exit 1
  fi
  if [[ ! -f "${VAST1_ENV_ACTIVATE}" ]]; then
    echo "VAST1_ENV_ACTIVATE does not exist: ${VAST1_ENV_ACTIVATE}" >&2
    exit 1
  fi
fi

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists. Attach with: tmux attach -t ${SESSION_NAME}" >&2
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
  cmd+=" && bash scripts/run.sh cambrian/main.py"

  local arg
  for arg in "${COMMON_ARGS[@]}"; do
    printf -v q_arg '%q' "${arg}"
    cmd+=" ${q_arg}"
  done
  for arg in "$@"; do
    printf -v q_arg '%q' "${arg}"
    cmd+=" ${q_arg}"
  done
  for arg in "${EXTRA_ARGS[@]}"; do
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

CMD_BASE_EYE="$(build_cmd 0 \
  "expname=vast1_tracking_base_eye" \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]")"

CMD_MULTI_EYE="$(build_cmd 0 \
  "expname=vast1_tracking_multi_eye" \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=multi_eye" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.num_eyes=[1,3]" \
  "env.agents.agent.eyes.eye.lon_range=[-30,30]" \
  "env.agents.agent.eyes.eye.lat_range=[-5,5]" \
  "env.agents.agent.eyes.eye.fov=[45,45]")"

CMD_OPTICS_EYE="$(build_cmd 1 \
  "expname=vast1_tracking_optics_eye" \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[45,45]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75")"

CMD_NARROW_LENS="$(build_cmd 1 \
  "expname=vast1_tracking_narrow_lens" \
  "env/agents@env.agents.agent=point" \
  "env/agents/eyes@env.agents.agent.eyes.eye=optics" \
  "env.agents.agent.eyes.eye.resolution=[20,20]" \
  "env.agents.agent.eyes.eye.fov=[15,15]" \
  "env.agents.agent.eyes.eye.aperture.radius=0.75")"

tmux new-session -d -s "${SESSION_NAME}" -n gpu0
tmux set-option -t "${SESSION_NAME}" remain-on-exit on
tmux set-option -t "${SESSION_NAME}" mouse on
tmux send-keys -t "${SESSION_NAME}:gpu0.0" "${CMD_BASE_EYE}" C-m
attach_pane_log "${SESSION_NAME}:gpu0.0" "${LOG_DIR}/gpu0_base_eye.log"
tmux split-window -h -t "${SESSION_NAME}:gpu0"
tmux send-keys -t "${SESSION_NAME}:gpu0.1" "${CMD_MULTI_EYE}" C-m
attach_pane_log "${SESSION_NAME}:gpu0.1" "${LOG_DIR}/gpu0_multi_eye.log"
tmux select-layout -t "${SESSION_NAME}:gpu0" tiled

tmux new-window -t "${SESSION_NAME}" -n gpu1
tmux send-keys -t "${SESSION_NAME}:gpu1.0" "${CMD_OPTICS_EYE}" C-m
attach_pane_log "${SESSION_NAME}:gpu1.0" "${LOG_DIR}/gpu1_optics_eye.log"
tmux split-window -h -t "${SESSION_NAME}:gpu1"
tmux send-keys -t "${SESSION_NAME}:gpu1.1" "${CMD_NARROW_LENS}" C-m
attach_pane_log "${SESSION_NAME}:gpu1.1" "${LOG_DIR}/gpu1_narrow_lens.log"
tmux select-layout -t "${SESSION_NAME}:gpu1" tiled

tmux select-window -t "${SESSION_NAME}:gpu0"

echo "Started tmux session '${SESSION_NAME}'."
if [[ -n "${TMUX:-}" ]]; then
  echo "Inside tmux already. Switch with: tmux switch-client -t ${SESSION_NAME}"
  echo "Or force attach with: unset TMUX; tmux attach -t ${SESSION_NAME}"
else
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
fi
echo "Pane logs: ${LOG_DIR}"
echo
echo "[GPU 0] base eye"
echo "  ${CMD_BASE_EYE}"
echo "[GPU 0] multi-eye"
echo "  ${CMD_MULTI_EYE}"
echo "[GPU 1] optics eye"
echo "  ${CMD_OPTICS_EYE}"
echo "[GPU 1] narrow lens"
echo "  ${CMD_NARROW_LENS}"
