#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SH="$REPO_ROOT/scripts/run.sh"
MAIN_PY="$REPO_ROOT/cambrian/main.py"

SESSION_NAME="${SESSION_NAME:-aci_baseline_control}"
BASELINE_ROOT="${BASELINE_ROOT:-$REPO_ROOT/logs/control_baseline}"
METRICS_CSV="${METRICS_CSV:-$BASELINE_ROOT/baseline_metrics.csv}"
LOCK_FILE="${LOCK_FILE:-$BASELINE_ROOT/.baseline_metrics.lock}"
MPLCONFIGDIR_ROOT="${MPLCONFIGDIR_ROOT:-/tmp/aci_mpl_baseline}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
SEED="${SEED:-0}"
GPU0_TASKS="${GPU0_TASKS:-detection tracking}"
GPU1_TASKS="${GPU1_TASKS:-navigation}"

mkdir -p "$BASELINE_ROOT" "$MPLCONFIGDIR_ROOT"

ensure_metrics_header() {
    if [[ -f "$METRICS_CSV" ]]; then
        return
    fi

    {
        flock 9
        if [[ ! -f "$METRICS_CSV" ]]; then
            printf '%s\n' \
                "timestamp_utc,task,physical_gpu,logdir,post_train_cumulative_reward,train_fitness,fov,resolution,num_eyes,lon_range,lat_range,total_timesteps,best_model_path" \
                > "$METRICS_CSV"
        fi
    } 9>>"$LOCK_FILE"
}

task_logdir() {
    local task="$1"
    printf '%s\n' "$BASELINE_ROOT/$task"
}

append_metrics_row() {
    local task="$1"
    local gpu="$2"
    local logdir="$3"

    {
        flock 9
        python - "$task" "$gpu" "$logdir" "$METRICS_CSV" "$TOTAL_TIMESTEPS" <<'PY'
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

task, gpu, logdir_arg, csv_arg, total_timesteps = sys.argv[1:]
logdir = Path(logdir_arg)
csv_path = Path(csv_arg)

config = yaml.safe_load((logdir / "config.yaml").read_text())
eye = config["env"]["agents"]["agent"]["eyes"]["eye"]

def read_float(path: Path):
    if not path.exists():
        return ""
    return float(path.read_text().strip())

row = {
    "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "task": task,
    "physical_gpu": gpu,
    "logdir": str(logdir),
    "post_train_cumulative_reward": read_float(logdir / "post_train_eval_fitness.txt"),
    "train_fitness": read_float(logdir / "train_fitness.txt"),
    "fov": json.dumps(eye.get("fov")),
    "resolution": json.dumps(eye.get("resolution")),
    "num_eyes": json.dumps(eye.get("num_eyes")),
    "lon_range": json.dumps(eye.get("lon_range")),
    "lat_range": json.dumps(eye.get("lat_range")),
    "total_timesteps": int(total_timesteps),
    "best_model_path": str(logdir / "best_model.zip"),
}

with csv_path.open("a", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=row.keys())
    writer.writerow(row)

print(
    "baseline|task={task}|gpu={gpu}|reward={reward}|resolution={resolution}|num_eyes={num_eyes}".format(
        task=row["task"],
        gpu=row["physical_gpu"],
        reward=row["post_train_cumulative_reward"],
        resolution=row["resolution"],
        num_eyes=row["num_eyes"],
    )
)
PY
    } 9>>"$LOCK_FILE"
}

run_task() {
    local gpu="$1"
    local task="$2"
    local logdir
    logdir="$(task_logdir "$task")"

    mkdir -p "$logdir"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] gpu=$gpu task=$task train -> $logdir"
    CUDA_VISIBLE_DEVICES="$gpu" \
    MPLCONFIGDIR="$MPLCONFIGDIR_ROOT/${task}_train_gpu${gpu}" \
    bash "$RUN_SH" "$MAIN_PY" --train \
        "example=$task" \
        "seed=$SEED" \
        "logdir=$logdir" \
        "trainer.total_timesteps=$TOTAL_TIMESTEPS"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] gpu=$gpu task=$task eval -> $logdir"
    CUDA_VISIBLE_DEVICES="$gpu" \
    MPLCONFIGDIR="$MPLCONFIGDIR_ROOT/${task}_eval_gpu${gpu}" \
    bash "$RUN_SH" "$MAIN_PY" --eval \
        "example=$task" \
        "seed=$SEED" \
        "logdir=$logdir" \
        "trainer/model=loaded_model" \
        "eval_env.n_eval_episodes=$EVAL_EPISODES" \
        "eval_env.save_filename=post_train_eval"

    append_metrics_row "$task" "$gpu" "$logdir"
}

worker_main() {
    local gpu="$1"
    shift

    if [[ "$#" -eq 0 ]]; then
        echo "No tasks assigned to gpu $gpu"
        return 0
    fi

    ensure_metrics_header

    for task in "$@"; do
        run_task "$gpu" "$task"
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
    cmd+="BASELINE_ROOT='$BASELINE_ROOT' "
    cmd+="METRICS_CSV='$METRICS_CSV' "
    cmd+="LOCK_FILE='$LOCK_FILE' "
    cmd+="MPLCONFIGDIR_ROOT='$MPLCONFIGDIR_ROOT' "
    cmd+="TOTAL_TIMESTEPS='$TOTAL_TIMESTEPS' "
    cmd+="EVAL_EPISODES='$EVAL_EPISODES' "
    cmd+="SEED='$SEED' "
    cmd+="bash '$SCRIPT_PATH' __worker__ '$gpu'"
    for task in "$@"; do
        cmd+=" '$task'"
    done

    if [[ "$target" == "new-session" ]]; then
        tmux new-session -d -s "$SESSION_NAME" -n "$window_name" "$cmd"
        return
    fi

    tmux new-window -t "$SESSION_NAME:" -n "$window_name" "$cmd"
}

main() {
    read -r -a gpu0_tasks <<< "$GPU0_TASKS"
    read -r -a gpu1_tasks <<< "$GPU1_TASKS"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists"
        exit 1
    fi

    ensure_metrics_header

    launch_tmux_worker new-session gpu0 0 "${gpu0_tasks[@]}"
    launch_tmux_worker new-window gpu1 1 "${gpu1_tasks[@]}"

    tmux set-option -t "$SESSION_NAME" remain-on-exit on >/dev/null

    cat <<EOF
Started tmux session: $SESSION_NAME
Baseline metrics: $METRICS_CSV

Workers:
  gpu0 -> ${gpu0_tasks[*]}
  gpu1 -> ${gpu1_tasks[*]}

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
