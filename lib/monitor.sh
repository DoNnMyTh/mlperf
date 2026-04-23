# shellcheck shell=bash
# Live training monitor — runs in background, tails $LOGDIR for MLlog
# JSON lines, flags divergence / step-time drift / eval-dominated wallclock.
#
# Emits alerts to $LOGDIR/monitor.alerts and optionally signals the parent
# pid when configured. Does NOT kill by itself unless MLPERF_MONITOR_KILL=1.

[[ -n "${_MLPERF_MONITOR_LOADED:-}" ]] && return 0
_MLPERF_MONITOR_LOADED=1

# Config knobs (override via env)
: "${MLPERF_MONITOR_ENABLE:=1}"
: "${MLPERF_MONITOR_INTERVAL:=30}"    # seconds between checks
: "${MLPERF_MONITOR_WARMUP_SKIP:=500}"# ignore first N steps
: "${MLPERF_MONITOR_SLOPE_WINDOW:=1000}" # steps for slope regression
: "${MLPERF_MONITOR_SLOPE_LIMIT:=0.0005}" # loss slope > this = divergence
: "${MLPERF_MONITOR_STEP_DRIFT:=1.3}" # observed/baseline > this = slow
: "${MLPERF_MONITOR_EVAL_BUDGET:=0.25}" # eval_time / train_time cap
: "${MLPERF_MONITOR_KILL:=0}"         # 1 = SIGTERM on divergence

# ----------------------------------------------------------------------
# mon_start logdir train_pid baseline_step_s
# Spawns a background watcher. Returns the watcher pid.
# ----------------------------------------------------------------------
mon_start() {
    local logdir="$1" train_pid="$2" baseline="${3:-0}"
    (( MLPERF_MONITOR_ENABLE == 1 )) || return 0
    mkdir -p "$logdir"
    local alert="$logdir/monitor.alerts"
    : > "$alert"

    (
        set +e
        local last_step=0 last_eval_s=0 cum_eval_s=0
        local start_ts; start_ts=$(date +%s)
        while kill -0 "$train_pid" 2>/dev/null; do
            sleep "$MLPERF_MONITOR_INTERVAL"

            # Scrape MLlog lines.
            local losses steps step_times
            losses=$(grep -rh '"reduced_train_loss"' "$logdir" 2>/dev/null \
                   | sed -nE 's/.*"reduced_train_loss"[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*/\1/p')
            steps=$(grep -rh '"step"' "$logdir" 2>/dev/null \
                   | sed -nE 's/.*"step"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p')
            step_times=$(grep -rh '"train_step_time"' "$logdir" 2>/dev/null \
                   | sed -nE 's/.*"train_step_time"[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*/\1/p')

            local cur_step; cur_step=$(printf '%s\n' "$steps" | tail -1)
            cur_step="${cur_step:-0}"
            [[ "$cur_step" -le "$MLPERF_MONITOR_WARMUP_SKIP" ]] && continue

            # --- 1. Divergence check (loss slope) ---
            local slope
            slope=$(printf '%s\n' "$losses" \
                   | tail -n "$MLPERF_MONITOR_SLOPE_WINDOW" \
                   | awk 'NR>10 {n++; sx+=NR; sy+=$1; sxx+=NR*NR; sxy+=NR*$1}
                          END{ if(n<20){print ""; exit}
                               denom=n*sxx-sx*sx
                               if (denom==0){print ""; exit}
                               printf "%.6f\n",(n*sxy-sx*sy)/denom }')
            if [[ -n "$slope" ]] && awk -v s="$slope" -v lim="$MLPERF_MONITOR_SLOPE_LIMIT" \
                   'BEGIN{exit !(s>lim)}'; then
                printf '[%s] DIVERGENCE step=%s slope=%s (limit=%s) — loss trending up\n' \
                    "$(date -Iseconds)" "$cur_step" "$slope" "$MLPERF_MONITOR_SLOPE_LIMIT" \
                    | tee -a "$alert" >&2
                if (( MLPERF_MONITOR_KILL == 1 )); then
                    printf '[%s] KILL sending SIGTERM to pid %s\n' \
                        "$(date -Iseconds)" "$train_pid" | tee -a "$alert" >&2
                    kill -TERM "$train_pid" 2>/dev/null
                    break
                fi
            fi

            # --- 2. Step-time drift ---
            if [[ "$baseline" != "0" && -n "$step_times" ]]; then
                local median
                median=$(printf '%s\n' "$step_times" | tail -100 | sort -g \
                       | awk '{a[NR]=$1} END{if(NR==0) exit; if(NR%2) print a[(NR+1)/2]; else printf "%.6f\n",(a[NR/2]+a[NR/2+1])/2}')
                if [[ -n "$median" ]] && awk -v m="$median" -v b="$baseline" \
                       -v lim="$MLPERF_MONITOR_STEP_DRIFT" \
                       'BEGIN{exit !(b>0 && m/b > lim)}'; then
                    printf '[%s] SLOW step=%s median=%ss baseline=%ss (%.2fx)\n' \
                        "$(date -Iseconds)" "$cur_step" "$median" "$baseline" \
                        "$(awk -v m="$median" -v b="$baseline" 'BEGIN{printf "%.2f",m/b}')" \
                        | tee -a "$alert" >&2
                fi
            fi

            # --- 3. Eval-dominated wallclock ---
            local now elapsed
            now=$(date +%s); elapsed=$((now - start_ts))
            # Rough: eval detection via "validation_loop" or "eval_start" lines.
            local eval_count
            eval_count=$(grep -rch '"eval_start"\|"validation_loop"' "$logdir" 2>/dev/null \
                       | awk -F':' '{s+=$NF} END{print s+0}')
            if (( eval_count > 0 && elapsed > 300 )); then
                local ratio
                ratio=$(awk -v e="$eval_count" -v es=70 -v tot="$elapsed" \
                       -v cap="$MLPERF_MONITOR_EVAL_BUDGET" 'BEGIN{
                           ev = e*es; tr = tot-ev; if (tr<=0) tr=1
                           r = ev/tot
                           printf "%.2f\n", r
                       }')
                if awk -v r="$ratio" -v cap="$MLPERF_MONITOR_EVAL_BUDGET" \
                       'BEGIN{exit !(r>cap)}'; then
                    printf '[%s] EVAL-HEAVY eval_count=%s ratio=%s cap=%s — raise VAL_CHECK_INTERVAL\n' \
                        "$(date -Iseconds)" "$eval_count" "$ratio" "$MLPERF_MONITOR_EVAL_BUDGET" \
                        | tee -a "$alert" >&2
                fi
            fi
        done
    ) &
    local mpid=$!
    echo "$mpid" > "$logdir/monitor.pid"
    info "Live monitor started (pid=$mpid, interval=${MLPERF_MONITOR_INTERVAL}s)."
    info "  Alerts: $alert"
    info "  Env knobs: MLPERF_MONITOR_KILL=$MLPERF_MONITOR_KILL  SLOPE_LIMIT=$MLPERF_MONITOR_SLOPE_LIMIT"
}

mon_stop() {
    local logdir="$1"
    local pidfile="$logdir/monitor.pid"
    [[ -f "$pidfile" ]] || return 0
    local mpid; mpid=$(cat "$pidfile")
    [[ -n "$mpid" ]] && kill "$mpid" 2>/dev/null || true
    rm -f "$pidfile"
}

# Quick one-shot report (for post-run summary).
mon_report() {
    local logdir="$1"
    local alert="$logdir/monitor.alerts"
    [[ -s "$alert" ]] || { info "Monitor: no alerts during run."; return; }
    warn "Monitor alerts during run:"
    sed 's/^/    /' "$alert" >&2
}
