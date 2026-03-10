#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SH="$REPO_ROOT/scripts/run.sh"
MAIN_PY="$REPO_ROOT/cambrian/main.py"

SESSION_NAME="${SESSION_NAME:-aci_stress_control}"
TASK="${TASK:-detection}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/logs/control_baseline/detection}"
STRESS_ROOT="${STRESS_ROOT:-$REPO_ROOT/logs/control_stress/$TASK}"
METRICS_CSV="${METRICS_CSV:-$STRESS_ROOT/stress_metrics.csv}"
LOCK_FILE="${LOCK_FILE:-$STRESS_ROOT/.stress_metrics.lock}"
MPLCONFIGDIR_ROOT="${MPLCONFIGDIR_ROOT:-/tmp/aci_mpl_stress}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-32}"
SEED="${SEED:-0}"
GPU_LIST="${GPU_LIST:-0 1}"
LIGHT_INTENSITIES="${LIGHT_INTENSITIES:-1.00 0.75 0.50 0.30 0.20 0.10 0.05 0.03 0.02 0.01}"

mkdir -p "$STRESS_ROOT" "$MPLCONFIGDIR_ROOT"

ensure_metrics_header() {
    if [[ -f "$METRICS_CSV" ]]; then
        return
    fi

    {
        flock 9
        if [[ ! -f "$METRICS_CSV" ]]; then
            printf '%s\n' \
                "timestamp_utc,task,physical_gpu,light_intensity,episodes,mean_reward,median_reward,std_reward,min_reward,max_reward,q25_reward,q75_reward,survival_rate_proxy,collapse_rate,eval_fitness,expdir,model_dir" \
                > "$METRICS_CSV"
        fi
    } 9>>"$LOCK_FILE"
}

intensity_tag() {
    local intensity="$1"
    local tag
    printf -v tag '%.2f' "$intensity"
    tag="${tag//./p}"
    printf '%s\n' "$tag"
}

stress_logdir() {
    local intensity="$1"
    printf '%s\n' "$STRESS_ROOT/intensity_$(intensity_tag "$intensity")"
}

light_overrides_for_task() {
    local task="$1"
    local intensity="$2"

    case "$task" in
        detection|tracking)
            printf '%s\n' \
                "env.agents.goal0.xml.overrides.0.mujoco.1.asset.1.material.2.emission=$intensity" \
                "env.agents.goal0.xml.overrides.0.mujoco.1.asset.2.material.2.emission=$intensity" \
                "env.agents.adversary0.xml.overrides.0.mujoco.1.asset.1.material.2.emission=$intensity" \
                "env.agents.adversary0.xml.overrides.0.mujoco.1.asset.2.material.2.emission=$intensity"
            ;;
        *)
            echo "Unsupported TASK=$task. This stress test currently supports detection and tracking." >&2
            return 1
            ;;
    esac
}

append_metrics_row() {
    local task="$1"
    local gpu="$2"
    local intensity="$3"
    local expdir="$4"

    {
        flock 9
        python - "$task" "$gpu" "$intensity" "$expdir" "$MODEL_DIR" "$METRICS_CSV" <<'PY'
import csv
import datetime as dt
import statistics
import sys
from pathlib import Path

task, gpu, intensity, expdir_arg, model_dir, csv_arg = sys.argv[1:]
expdir = Path(expdir_arg)
csv_path = Path(csv_arg)
monitor_path = expdir / "eval_monitor.csv"

rewards = []
with monitor_path.open() as handle:
    for line in handle:
        if line.startswith("#"):
            continue
        row = line.strip().split(",")
        if row[0] == "r":
            continue
        rewards.append(float(row[0]))

if not rewards:
    raise SystemExit(f"no rewards found in {monitor_path}")

rewards_sorted = sorted(rewards)

def quantile(values, q):
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight

mean_reward = statistics.fmean(rewards)
median_reward = statistics.median(rewards)
std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
survival_rate_proxy = sum(reward > 0.0 for reward in rewards) / len(rewards)
collapse_rate = 1.0 - survival_rate_proxy
eval_fitness_path = expdir / "stress_eval_fitness.txt"
eval_fitness = float(eval_fitness_path.read_text().strip()) if eval_fitness_path.exists() else ""

row = {
    "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "task": task,
    "physical_gpu": gpu,
    "light_intensity": float(intensity),
    "episodes": len(rewards),
    "mean_reward": mean_reward,
    "median_reward": median_reward,
    "std_reward": std_reward,
    "min_reward": min(rewards),
    "max_reward": max(rewards),
    "q25_reward": quantile(rewards_sorted, 0.25),
    "q75_reward": quantile(rewards_sorted, 0.75),
    "survival_rate_proxy": survival_rate_proxy,
    "collapse_rate": collapse_rate,
    "eval_fitness": eval_fitness,
    "expdir": str(expdir),
    "model_dir": model_dir,
}

with csv_path.open("a", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=row.keys())
    writer.writerow(row)

print(
    "stress|task={task}|gpu={gpu}|light_intensity={light}|episodes={episodes}|mean_reward={mean:.4f}|survival_rate_proxy={survival:.4f}|collapse_rate={collapse:.4f}".format(
        task=row["task"],
        gpu=row["physical_gpu"],
        light=row["light_intensity"],
        episodes=row["episodes"],
        mean=row["mean_reward"],
        survival=row["survival_rate_proxy"],
        collapse=row["collapse_rate"],
    )
)
PY
    } 9>>"$LOCK_FILE"
}

run_intensity() {
    local gpu="$1"
    local intensity="$2"
    local expdir
    expdir="$(stress_logdir "$intensity")"

    mkdir -p "$expdir"

    mapfile -t light_overrides < <(light_overrides_for_task "$TASK" "$intensity")

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] gpu=$gpu task=$TASK light_intensity=$intensity eval -> $expdir"

    CUDA_VISIBLE_DEVICES="$gpu" \
    MPLCONFIGDIR="$MPLCONFIGDIR_ROOT/${TASK}_gpu${gpu}_$(intensity_tag "$intensity")" \
    bash "$RUN_SH" "$MAIN_PY" --eval \
        "example=$TASK" \
        "seed=$SEED" \
        "logdir=$expdir" \
        "trainer/model=loaded_model" \
        "trainer.model.path=$MODEL_DIR/best_model" \
        "eval_env.n_eval_episodes=$N_EVAL_EPISODES" \
        "eval_env.save_filename=stress_eval" \
        "~eval_env.step_fn.respawn_objects_if_agent_close" \
        "~eval_env.reward_fn.reward_if_goal_respawned" \
        "~eval_env.reward_fn.penalize_if_adversary_respawned" \
        "eval_env.truncation_fn.truncate_if_close_to_adversary.disable=false" \
        "eval_env.termination_fn.terminate_if_close_to_goal.disable=false" \
        "${light_overrides[@]}"

    append_metrics_row "$TASK" "$gpu" "$intensity" "$expdir"
}

worker_main() {
    local gpu="$1"
    shift

    if [[ "$#" -eq 0 ]]; then
        echo "No light intensities assigned to gpu $gpu"
        return 0
    fi

    ensure_metrics_header

    for intensity in "$@"; do
        run_intensity "$gpu" "$intensity"
    done
}

launch_tmux_worker() {
    local target="$1"
    local window_name="$2"
    local gpu="$3"
    shift 3

    local cmd
    cmd="cd '$REPO_ROOT' && "
    cmd+="SESSION_NAME='$SESSION_NAME' "
    cmd+="TASK='$TASK' "
    cmd+="MODEL_DIR='$MODEL_DIR' "
    cmd+="STRESS_ROOT='$STRESS_ROOT' "
    cmd+="METRICS_CSV='$METRICS_CSV' "
    cmd+="LOCK_FILE='$LOCK_FILE' "
    cmd+="MPLCONFIGDIR_ROOT='$MPLCONFIGDIR_ROOT' "
    cmd+="N_EVAL_EPISODES='$N_EVAL_EPISODES' "
    cmd+="SEED='$SEED' "
    cmd+="bash '$SCRIPT_PATH' __worker__ '$gpu'"
    for intensity in "$@"; do
        cmd+=" '$intensity'"
    done

    if [[ "$target" == "new-session" ]]; then
        tmux new-session -d -s "$SESSION_NAME" -n "$window_name" "$cmd"
        return
    fi

    tmux new-window -t "$SESSION_NAME:" -n "$window_name" "$cmd"
}

main() {
    if [[ ! -f "$MODEL_DIR/best_model.zip" ]]; then
        echo "Missing model checkpoint: $MODEL_DIR/best_model.zip" >&2
        exit 1
    fi

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists"
        exit 1
    fi

    ensure_metrics_header

    read -r -a gpus <<< "$GPU_LIST"
    read -r -a intensities <<< "$LIGHT_INTENSITIES"

    if [[ "${#gpus[@]}" -eq 0 ]]; then
        echo "GPU_LIST must contain at least one GPU id" >&2
        exit 1
    fi

    local -a assignments
    local idx
    for idx in "${!gpus[@]}"; do
        assignments[idx]=""
    done

    for idx in "${!intensities[@]}"; do
        local gpu_index=$((idx % ${#gpus[@]}))
        assignments[gpu_index]="${assignments[gpu_index]} ${intensities[idx]}"
    done

    local first=1
    for idx in "${!gpus[@]}"; do
        read -r -a assigned <<< "${assignments[idx]}"
        if [[ "${#assigned[@]}" -eq 0 ]]; then
            continue
        fi
        if [[ "$first" -eq 1 ]]; then
            launch_tmux_worker new-session "gpu${gpus[idx]}" "${gpus[idx]}" "${assigned[@]}"
            first=0
        else
            launch_tmux_worker new-window "gpu${gpus[idx]}" "${gpus[idx]}" "${assigned[@]}"
        fi
    done

    tmux set-option -t "$SESSION_NAME" remain-on-exit on >/dev/null

    cat <<EOF
Started tmux session: $SESSION_NAME
Stress metrics: $METRICS_CSV
Model dir: $MODEL_DIR
Task: $TASK
Light intensities: ${intensities[*]}

Inspect:
  tmux attach -t $SESSION_NAME
  tail -f $METRICS_CSV
EOF
}

if [[ "${1:-}" == "__worker__" ]]; then
    shift
    worker_main "$@"
    exit 0
fi

main "$@"
