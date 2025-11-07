#!/usr/bin/env bash
# NOTE: make it executable $chmod +x free_jetson_memory.sh

set -euo pipefail

bold() { echo -e "\e[1m$*\e[0m"; }
warn() { echo -e "\e[33m$*\e[0m"; }

DRY_RUN=0
PRUNE_VOLUMES=0
DEEP=0      # deep clean of /tmp and /dev/shm, off by default
KEEP_IMAGES=1
KEEP_OLLAMA_STOPPED=0  # by default, restart ollama after cleanup
OLLAMA_WAS_STOPPED=0

# User-tunable list of process patterns to kill if found.
DEFAULT_KILL_PATTERNS=(
  "ollama serve"
)

CONFIG_DIR="${HOME}/.config/jetson-demo-prep"
KILLLIST_FILE="${CONFIG_DIR}/killlist.txt"

ensure_config() {
  mkdir -p "$CONFIG_DIR"
  if [[ ! -f "$KILLLIST_FILE" ]]; then
    printf "%s\n" "${DEFAULT_KILL_PATTERNS[@]}" > "$KILLLIST_FILE"
  fi
}

require_root() {
  if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo:"
    echo "  sudo $0 $*"
    exit 1
  fi
}

mem_total_kB() { awk '/MemTotal:/ {print $2}' /proc/meminfo; }
mem_available_kB() { awk '/MemAvailable:/ {print $2}' /proc/meminfo; }

show_mem() {
  bold "[RAM] $(date '+%F %T')"
  free -h
  echo
}

stop_service_if_active() {
  local svc="$1"
  if systemctl is-active --quiet "$svc"; then
    run_cmd systemctl stop "$svc"
    if [[ "$svc" == "ollama" ]]; then
      OLLAMA_WAS_STOPPED=1
    fi
    return 0
  fi
  return 1
}

run_cmd() {
  if (( DRY_RUN )); then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

kill_patterns() {
  local pattern
  local me=$$
  while IFS= read -r pattern || [[ -n "$pattern" ]]; do
    [[ -z "$pattern" || "$pattern" =~ ^# ]] && continue
    if pgrep -fa "$pattern" >/dev/null 2>&1; then
      bold "[kill] $pattern"
      # Try TERM then KILL, avoid killing this script or its parent shell
      for pid in $(pgrep -f "$pattern" || true); do
        [[ "$pid" == "$me" ]] && continue
        [[ "$pid" == "$PPID" ]] && continue
        run_cmd kill -TERM "$pid" || true
      done
      sleep 1
      for pid in $(pgrep -f "$pattern" || true); do
        [[ "$pid" == "$me" ]] && continue
        [[ "$pid" == "$PPID" ]] && continue
        run_cmd kill -KILL "$pid" || true
      done
    fi
  done < "$KILLLIST_FILE"
}

docker_cleanup() {
  if ! systemctl is-active --quiet docker; then
    run_cmd systemctl start docker || true
    local started=1
  else
    local started=0
  fi

  if command -v docker >/dev/null 2>&1; then
    bold "[docker] stopping running containers"
    docker ps -q | xargs -r -I{} bash -c '[[ '"$DRY_RUN"' -eq 1 ]] && echo "[dry-run] docker stop {}" || docker stop {}' || true

    bold "[docker] pruning stopped containers and unused networks"
    if (( DRY_RUN )); then
      echo "[dry-run] docker container prune -f"
      echo "[dry-run] docker network prune -f"
      if (( PRUNE_VOLUMES )); then echo "[dry-run] docker volume prune -f"; fi
      if (( KEEP_IMAGES )); then
        echo "[dry-run] docker image prune -f"
      else
        echo "[dry-run] docker image prune -f -a"
      fi
      echo "[dry-run] docker builder prune -af"
    else
      docker container prune -f || true
      docker network prune -f || true
      if (( PRUNE_VOLUMES )); then docker volume prune -f || true; fi
      if (( KEEP_IMAGES )); then
        # Only remove dangling images (untagged)
        docker image prune -f || true
      else
        # Remove all unused images
        docker image prune -f -a || true
      fi
      docker builder prune -af || true
    fi
  fi

  if (( started )); then
    run_cmd systemctl stop docker || true
  fi
}

drop_caches_and_swap_reset() {
  bold "[vm] syncing and dropping caches"
  run_cmd sync
  # 3 drops pagecache, dentries, inodes
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    if (( DRY_RUN )); then
      echo "[dry-run] echo 3 > /proc/sys/vm/drop_caches"
    else
      echo 3 > /proc/sys/vm/drop_caches || true
    fi
  fi

  bold "[swap] cycle swap"
  run_cmd swapoff -a || true
  sleep 1
  run_cmd swapon -a || true
}

camera_stack_restart() {
  if systemctl list-unit-files | grep -q '^nvargus-daemon\.service'; then
    bold "[camera] restarting nvargus-daemon"
    run_cmd systemctl restart nvargus-daemon || true
  fi
}

deep_clean_tmp() {
  if (( DEEP )); then
    warn "[deep] cleaning /dev/shm and stale tmp files"
    # Be careful in desktop mode
    run_cmd bash -c 'find /dev/shm -mindepth 1 -maxdepth 1 -exec rm -rf {} +'
    run_cmd bash -c 'find /tmp -maxdepth 1 -mtime +1 -exec rm -rf {} +'
    journalctl --disk-usage >/dev/null 2>&1 || true
    run_cmd journalctl --vacuum-time=2d >/dev/null 2>&1 || true
  fi
}

fix_duplicate_nvpmodel() {
  # Sometimes nvpmodel_indicator spawns many duplicate processes
  local count
  count=$(pgrep -fc nvpmodel_indicator 2>/dev/null || echo "0")
  # Ensure count is a valid number
  count=${count//[^0-9]/}
  count=${count:-0}
  
  if (( count > 2 )); then
    warn "[nvidia] found $count nvpmodel_indicator processes (should be 1-2), cleaning up"
    run_cmd killall nvpmodel_indicator || true
    sleep 1
    # Restart one instance
    if (( ! DRY_RUN )); then
      nohup python3 /usr/share/nvpmodel_indicator/nvpmodel_indicator.py >/dev/null 2>&1 &
      disown
    else
      echo "[dry-run] nohup python3 /usr/share/nvpmodel_indicator/nvpmodel_indicator.py >/dev/null 2>&1 &"
    fi
  fi
}

largest_processes() {
  bold "[top RSS] largest processes"
  ps -eo pid,cmd,%mem,rss --sort=-rss | awk 'NR==1 || NR<=16 {printf "%-7s %-6s %-7s %s\n",$1,$3"%",$4"kB",$2}' || true
  echo
}

usage() {
  cat <<EOF
Usage: sudo jetson-demo-scrub.sh [options]

Options:
  --dry-run           Show actions without doing them
  --prune-volumes     Also prune unused Docker volumes
  --deep              Extra cleanup of /dev/shm and stale /tmp (safe but cautious)
  --purge-images      Also prune Docker images (will re-pull later). Default keeps images.
  --stop-ollama       Keep Ollama stopped after cleanup (default: auto-restart)
  --show-killlist     Print current kill patterns file
  --edit-killlist     Open killlist in \$EDITOR

Kill patterns file:
  ${KILLLIST_FILE}
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run) DRY_RUN=1 ;;
      --prune-volumes) PRUNE_VOLUMES=1 ;;
      --deep) DEEP=1 ;;
      --purge-images) KEEP_IMAGES=0 ;;
      --stop-ollama) KEEP_OLLAMA_STOPPED=1 ;;
      --show-killlist) ensure_config; cat "$KILLLIST_FILE"; exit 0 ;;
      --edit-killlist) ensure_config; ${EDITOR:-nano} "$KILLLIST_FILE"; exit 0 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
  done
}

main() {
  require_root
  ensure_config
  parse_args "$@"

  bold "Cleanup starting"
  largest_processes
  show_mem
  local before_avail=$(mem_available_kB)

  # 1) Stop Ollama and kill common demo leftovers
  stop_service_if_active ollama || true
  kill_patterns

  # 2) Clean Docker without losing images by default
  docker_cleanup

  # 3) Release FS cache and swap
  drop_caches_and_swap_reset

  # 4) Release Argus buffers
  camera_stack_restart

  # 5) Optional deeper temp cleanup
  deep_clean_tmp

  # 6) Fix duplicate nvpmodel_indicator processes
  fix_duplicate_nvpmodel

  # 7) Restart Ollama automatically (unless --stop-ollama was used)
  if (( OLLAMA_WAS_STOPPED && ! KEEP_OLLAMA_STOPPED )); then
    bold "[ollama] restarting service"
    run_cmd systemctl start ollama
    sleep 2  # give it a moment to start
    OLLAMA_WAS_STOPPED=0
  fi

  # 8) Report
  show_mem
  largest_processes
  local after_avail=$(mem_available_kB)
  local reclaimed_kB=$(( after_avail - before_avail ))
  bold "[reclaimed] $((reclaimed_kB/1024)) MB approx"
  
  # If Ollama was kept stopped
  if (( OLLAMA_WAS_STOPPED )); then
    echo
    warn "Ollama is stopped. To restart:"
    echo "    sudo systemctl start ollama"
  fi
  
  echo
  bold "Done."
}

main "$@"
