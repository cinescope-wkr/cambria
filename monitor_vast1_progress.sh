#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE_DIR="${1:-$(date -u +%Y-%m-%d)}"
BASE_DIR="${ROOT_DIR}/logs/${DATE_DIR}"

EXPERIMENTS=(
  "vast1_tracking_base_eye"
  "vast1_tracking_multi_eye"
  "vast1_tracking_optics_eye"
  "vast1_tracking_narrow_lens"
)

resolve_expdir() {
  local exp="$1"
  local candidate="${BASE_DIR}/${exp}"
  if [[ -d "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return
  fi

  local latest
  latest="$(find "${ROOT_DIR}/logs" -type d -name "${exp}" | sort | tail -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
    return
  fi

  printf '%s\n' "${candidate}"
}

latest_checkpoint_step() {
  local checkpoint_dir="$1"
  if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "-"
    return
  fi

  local latest
  latest="$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'policy_*.pt' | sort -V | tail -n 1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "-"
    return
  fi

  basename "${latest}" | sed -E 's/^policy_([0-9]+)\.pt$/\1/'
}

latest_eval_reward() {
  local eval_csv="$1"
  if [[ ! -f "${eval_csv}" ]]; then
    echo "-"
    return
  fi

  awk -F, 'NR>2{last=$1} END{if (last=="") print "-"; else print last}' "${eval_csv}"
}

status_flag() {
  local expdir="$1"
  if [[ -f "${expdir}/finished" ]]; then
    echo "finished"
  elif [[ -f "${expdir}/best_model.zip" ]]; then
    echo "running"
  elif [[ -f "${expdir}/config.yaml" ]]; then
    echo "started"
  else
    echo "missing"
  fi
}

echo "Requested base dir: ${BASE_DIR}"
printf "%-28s %-10s %-12s %-12s %-10s\n" "experiment" "status" "ckpt_step" "last_eval_r" "best_model"
printf "%-28s %-10s %-12s %-12s %-10s\n" "----------" "------" "---------" "-----------" "----------"

for exp in "${EXPERIMENTS[@]}"; do
  expdir="$(resolve_expdir "${exp}")"
  checkpoint_step="$(latest_checkpoint_step "${expdir}/checkpoints")"
  eval_reward="$(latest_eval_reward "${expdir}/eval_monitor.csv")"
  status="$(status_flag "${expdir}")"
  best_model="no"
  if [[ -f "${expdir}/best_model.zip" ]]; then
    best_model="yes"
  fi

  printf "%-28s %-10s %-12s %-12s %-10s\n" "${exp}" "${status}" "${checkpoint_step}" "${eval_reward}" "${best_model}"
done

echo
echo "Recent stderr tails:"
for exp in "${EXPERIMENTS[@]}"; do
  expdir="$(resolve_expdir "${exp}")"
  errlog="${expdir}/logs/err.log"
  echo "== ${exp} =="
  if [[ -f "${errlog}" ]]; then
    tail -n 5 "${errlog}"
  else
    echo "(no err.log yet)"
  fi
done
