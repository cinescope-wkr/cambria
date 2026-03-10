#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-/tmp}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

SMOKE_EXPNAME="${SMOKE_EXPNAME:-smoke_optics_morphogenesis}"
SMOKE_TASK="${SMOKE_TASK:-tracking}"
SMOKE_TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-2000}"
SMOKE_EVAL_FREQ="${SMOKE_EVAL_FREQ:-500}"
SMOKE_POPULATION_SIZE="${SMOKE_POPULATION_SIZE:-2}"
SMOKE_NUM_GENERATIONS="${SMOKE_NUM_GENERATIONS:-1}"
SMOKE_EVAL_EPISODES="${SMOKE_EVAL_EPISODES:-2}"
SMOKE_HYDRA_LAUNCHER="${SMOKE_HYDRA_LAUNCHER:-basic}"

echo "Running optics morphogenesis smoke test..."
echo "  expname:      ${SMOKE_EXPNAME}"
echo "  task:         ${SMOKE_TASK}"
echo "  timesteps:    ${SMOKE_TOTAL_TIMESTEPS}"
echo "  eval_freq:    ${SMOKE_EVAL_FREQ}"
echo "  population:   ${SMOKE_POPULATION_SIZE}"
echo "  generations:  ${SMOKE_NUM_GENERATIONS}"
echo "  eval episodes:${SMOKE_EVAL_EPISODES}"
echo "  launcher:     ${SMOKE_HYDRA_LAUNCHER}"

bash scripts/run.sh cambrian/main.py --train -m \
  task="${SMOKE_TASK}" \
  expname="${SMOKE_EXPNAME}" \
  evo=optics_morphogenesis \
  trainer.total_timesteps="${SMOKE_TOTAL_TIMESTEPS}" \
  trainer.callbacks.eval_callback.eval_freq="${SMOKE_EVAL_FREQ}" \
  trainer.callbacks.eval_callback.render=false \
  trainer.n_envs=1 \
  env.n_eval_episodes="${SMOKE_EVAL_EPISODES}" \
  eval_env.n_eval_episodes="${SMOKE_EVAL_EPISODES}" \
  evo.population_size="${SMOKE_POPULATION_SIZE}" \
  evo.num_generations="${SMOKE_NUM_GENERATIONS}" \
  hydra/launcher="${SMOKE_HYDRA_LAUNCHER}" \
  trainer/fitness_penalty_fn=optics_morphology \
  env/agents@env.agents.agent=point \
  env/agents/eyes@env.agents.agent.eyes.eye=optics \
  env/renderer=renderer \
  env.add_overlays=false \
  env.debug_overlays_size=0 \
  env.agents.agent.eyes.eye.scale_intensity=True \
  env.agents.agent.eyes.eye.noise_std=0.03

TODAY="$(date -u +%F)"
BASE_DIR="$ROOT_DIR/logs/${TODAY}/${SMOKE_EXPNAME}"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Smoke test failed: log directory not found: $BASE_DIR" >&2
  exit 1
fi

echo
echo "Verifying outputs in $BASE_DIR"

missing=0
while IFS= read -r expdir; do
  echo "--- ${expdir#$ROOT_DIR/}"
  for required in config.yaml train_fitness.txt train_fitness_base.txt train_fitness_penalty.txt finished; do
    if [[ -f "$expdir/$required" ]]; then
      echo "ok   $required"
    else
      echo "MISS $required" >&2
      missing=1
    fi
  done
  if [[ -f "$expdir/train_fitness_penalty.txt" ]]; then
    echo "penalty=$(cat "$expdir/train_fitness_penalty.txt")"
  fi
done < <(find "$BASE_DIR" -mindepth 2 -maxdepth 2 -type d -name 'rank_*' | sort)

if [[ "$missing" -ne 0 ]]; then
  echo "Smoke test failed: one or more expected artifacts were missing." >&2
  exit 1
fi

echo
echo "Smoke test completed successfully."
echo "Logs: $BASE_DIR"
