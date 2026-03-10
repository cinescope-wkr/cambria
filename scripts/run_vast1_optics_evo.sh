#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-optics_evo_divergent}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT_DIR}/logs/vast1_runs"
RESUME_ROOT="${ROOT_DIR}/logs/optics_evo_runs"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
PANE_LOG_DIR="${LOG_ROOT}/${RUN_STAMP}"
VAST1_ENV_ACTIVATE="${VAST1_ENV_ACTIVATE:-}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
POPULATION_SIZE="${POPULATION_SIZE:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-4}"
HYDRA_LAUNCHER="${HYDRA_LAUNCHER:-basic}"

SHALLOW_EXPNAME="${SHALLOW_EXPNAME:-optics_evo_shallow_sea}"
DEEP_EXPNAME="${DEEP_EXPNAME:-optics_evo_deep_sea}"
SHALLOW_NOISE_STD="${SHALLOW_NOISE_STD:-0.01}"
DEEP_NOISE_STD="${DEEP_NOISE_STD:-0.1}"

EXTRA_ARGS=("$@")

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found in PATH." >&2
  exit 1
fi

if [[ -n "${VAST1_ENV_ACTIVATE}" ]]; then
  if [[ "${VAST1_ENV_ACTIVATE}" == "/path/to/venv/bin/activate" ]]; then
    echo "VAST1_ENV_ACTIVATE is still set to the placeholder path." >&2
    exit 1
  fi
  if [[ ! -f "${VAST1_ENV_ACTIVATE}" ]]; then
    echo "VAST1_ENV_ACTIVATE does not exist: ${VAST1_ENV_ACTIVATE}" >&2
    exit 1
  fi
fi

mkdir -p "${PANE_LOG_DIR}"

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

  cmd+=" && export MUJOCO_GL=egl PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${gpu}"
  cmd+=" && bash scripts/run.sh cambrian/main.py"

  local args=(
    "--train"
    "-m"
    "task=tracking"
    "evo=optics_morphogenesis"
    "trainer/fitness_penalty_fn=optics_morphology"
    "hydra/launcher=${HYDRA_LAUNCHER}"
    "trainer.total_timesteps=${TOTAL_TIMESTEPS}"
    "trainer.callbacks.eval_callback.eval_freq=${EVAL_FREQ}"
    "trainer.callbacks.eval_callback.render=false"
    "trainer.n_envs=1"
    "env.n_eval_episodes=${N_EVAL_EPISODES}"
    "eval_env.n_eval_episodes=${N_EVAL_EPISODES}"
    "evo.population_size=${POPULATION_SIZE}"
    "evo.num_generations=${NUM_GENERATIONS}"
    "env/agents@env.agents.agent=point"
    "env/agents/eyes@env.agents.agent.eyes.eye=optics"
    "env/renderer=renderer"
    "env.add_overlays=false"
    "env.debug_overlays_size=0"
    "env.agents.agent.eyes.eye.scale_intensity=True"
  )

  local arg
  for arg in "${args[@]}"; do
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

CMD_SHALLOW="$(build_cmd 0 \
  "expname=${SHALLOW_EXPNAME}" \
  "logdir=${RESUME_ROOT}/${SHALLOW_EXPNAME}" \
  "env.agents.agent.eyes.eye.noise_std=${SHALLOW_NOISE_STD}")"

CMD_DEEP="$(build_cmd 1 \
  "expname=${DEEP_EXPNAME}" \
  "logdir=${RESUME_ROOT}/${DEEP_EXPNAME}" \
  "env.agents.agent.eyes.eye.noise_std=${DEEP_NOISE_STD}")"

tmux new-session -d -s "${SESSION_NAME}" -n shallow
tmux set-option -t "${SESSION_NAME}" remain-on-exit on
tmux set-option -t "${SESSION_NAME}" mouse on
tmux send-keys -t "${SESSION_NAME}:shallow.0" "${CMD_SHALLOW}" C-m
attach_pane_log "${SESSION_NAME}:shallow.0" "${PANE_LOG_DIR}/gpu0_shallow_sea.log"

tmux new-window -t "${SESSION_NAME}" -n deep
tmux send-keys -t "${SESSION_NAME}:deep.0" "${CMD_DEEP}" C-m
attach_pane_log "${SESSION_NAME}:deep.0" "${PANE_LOG_DIR}/gpu1_deep_sea.log"

tmux select-window -t "${SESSION_NAME}:shallow"

echo "Started tmux session '${SESSION_NAME}'."
if [[ -n "${TMUX:-}" ]]; then
  echo "Inside tmux already. Switch with: tmux switch-client -t ${SESSION_NAME}"
  echo "Or force attach with: unset TMUX; tmux attach -t ${SESSION_NAME}"
else
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
fi
echo "Pane logs: ${PANE_LOG_DIR}"
echo
echo "[GPU 0] shallow sea"
echo "  ${CMD_SHALLOW}"
echo "  resume dir: ${RESUME_ROOT}/${SHALLOW_EXPNAME}"
echo "[GPU 1] deep sea"
echo "  ${CMD_DEEP}"
echo "  resume dir: ${RESUME_ROOT}/${DEEP_EXPNAME}"
