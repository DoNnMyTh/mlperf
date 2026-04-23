# shellcheck shell=bash
# Hardware inventory + calibration probe + dynamic step-time cache.
# Sourced by mlperf.sh. Depends on lib/common.sh (info/warn/err/die).
#
# Cache schema (TSV, one row per measured (hw,config)):
#   key<TAB>step_s<TAB>mem_gb<TAB>loss_slope<TAB>ts
# key = gpu_model|ngpu|tp|pp|cp|fp8|microbs|wl

[[ -n "${_MLPERF_CALIBRATE_LOADED:-}" ]] && return 0
_MLPERF_CALIBRATE_LOADED=1

: "${MLPERF_CACHE_DIR:=${XDG_CACHE_HOME:-$HOME/.cache}/mlperf}"
CAL_CACHE="$MLPERF_CACHE_DIR/calibration.tsv"
CAL_HW_CACHE="$MLPERF_CACHE_DIR/hw_inventory.env"

# ----------------------------------------------------------------------
# HW inventory — populates CAL_* globals. Cached in hw_inventory.env so
# repeat runs on same host skip the nvidia-smi / topo calls.
# ----------------------------------------------------------------------
cal_inventory_hw() {
    CAL_GPU_MODEL=""
    CAL_GPU_ARCH=""
    CAL_GPU_COUNT=0
    CAL_GPU_MEM_GB=0
    CAL_NVLINK="unknown"
    CAL_CPU_COUNT=0
    CAL_HOST_MEM_GB=0
    CAL_DATADIR_FS=""

    if [[ -f "$CAL_HW_CACHE" && -z "${MLPERF_CAL_REFRESH:-}" ]]; then
        # shellcheck disable=SC1090
        source "$CAL_HW_CACHE" 2>/dev/null || true
        [[ -n "$CAL_GPU_MODEL" && "$CAL_GPU_COUNT" -gt 0 ]] && return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local smi
        smi=$(nvidia-smi --query-gpu=name,compute_cap,memory.total \
              --format=csv,noheader,nounits 2>/dev/null) || smi=""
        if [[ -n "$smi" ]]; then
            CAL_GPU_COUNT=$(printf '%s\n' "$smi" | wc -l | tr -d ' ')
            CAL_GPU_MODEL=$(printf '%s\n' "$smi" | head -1 | awk -F',' '{gsub(/^ +| +$/,"",$1); print $1}')
            local cc
            cc=$(printf '%s\n' "$smi" | head -1 | awk -F',' '{gsub(/[^0-9.]/,"",$2); print $2}')
            CAL_GPU_ARCH=$(awk -v c="$cc" 'BEGIN{gsub(/\./,"",c); print c+0}')
            CAL_GPU_MEM_GB=$(printf '%s\n' "$smi" | head -1 | awk -F',' '{gsub(/[^0-9]/,"",$3); print int($3/1024)}')
        fi
        # NVLink topology — any NV* = NVLink; PIX/SYS = PCIe only.
        local topo
        topo=$(nvidia-smi topo -m 2>/dev/null) || topo=""
        if [[ -n "$topo" ]]; then
            if grep -qE 'NV[0-9]+' <<<"$topo"; then CAL_NVLINK="nvlink"
            elif grep -q 'PIX\|PXB' <<<"$topo";  then CAL_NVLINK="pcie"
            else                                     CAL_NVLINK="mixed"
            fi
        fi
    fi

    CAL_CPU_COUNT=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)
    if [[ -r /proc/meminfo ]]; then
        CAL_HOST_MEM_GB=$(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo)
    fi
    if [[ -n "${DATADIR:-}" && -d "${DATADIR}" ]]; then
        CAL_DATADIR_FS=$(stat -f -c '%T' "$DATADIR" 2>/dev/null \
                      || df -T "$DATADIR" 2>/dev/null | awk 'NR==2{print $2}' \
                      || echo unknown)
    fi

    mkdir -p "$MLPERF_CACHE_DIR"
    cat > "$CAL_HW_CACHE" <<EOF
CAL_GPU_MODEL='$CAL_GPU_MODEL'
CAL_GPU_ARCH='$CAL_GPU_ARCH'
CAL_GPU_COUNT='$CAL_GPU_COUNT'
CAL_GPU_MEM_GB='$CAL_GPU_MEM_GB'
CAL_NVLINK='$CAL_NVLINK'
CAL_CPU_COUNT='$CAL_CPU_COUNT'
CAL_HOST_MEM_GB='$CAL_HOST_MEM_GB'
CAL_DATADIR_FS='$CAL_DATADIR_FS'
EOF
}

cal_print_hw() {
    cal_inventory_hw
    say "Hardware inventory"
    info "GPU       : ${CAL_GPU_COUNT}× ${CAL_GPU_MODEL:-unknown} (sm_${CAL_GPU_ARCH:-?}, ${CAL_GPU_MEM_GB}GB each)"
    info "NVLink    : ${CAL_NVLINK}"
    info "CPU/Mem   : ${CAL_CPU_COUNT} cores / ${CAL_HOST_MEM_GB}GB host RAM"
    info "DATADIR FS: ${CAL_DATADIR_FS:-not-set}"
}

# ----------------------------------------------------------------------
# Cache primitives
# ----------------------------------------------------------------------
cal_key() {
    # wl, ngpu, tp, pp, cp, fp8, microbs
    printf '%s|%s|%s|%s|%s|%s|%s|%s' \
        "${CAL_GPU_MODEL// /_}" "$2" "$3" "$4" "$5" "$6" "$7" "$1"
}

cal_cache_lookup() {
    # args: wl ngpu tp pp cp fp8 microbs
    # stdout: "step_s<TAB>mem_gb<TAB>loss_slope" or empty
    local k; k=$(cal_key "$@")
    [[ -f "$CAL_CACHE" ]] || return 1
    awk -F'\t' -v k="$k" '$1==k {print $2"\t"$3"\t"$4; found=1; exit}
                          END{exit !found}' "$CAL_CACHE"
}

_cal_cache_record_unlocked() {
    local k="$1" step="$2" mem="$3" slope="$4"
    # Per-process tmp prevents concurrent writers clobbering each other's
    # scratch file even before the flock has been acquired.
    local tmp="$CAL_CACHE.tmp.$$"
    [[ -f "$CAL_CACHE" ]] && awk -F'\t' -v k="$k" '$1!=k' "$CAL_CACHE" > "$tmp" || : > "$tmp"
    printf '%s\t%s\t%s\t%s\t%s\n' "$k" "$step" "$mem" "$slope" "$(date -Iseconds)" >> "$tmp"
    mv "$tmp" "$CAL_CACHE"
}

cal_cache_record() {
    # args: wl ngpu tp pp cp fp8 microbs step_s mem_gb loss_slope
    local wl="$1" ngpu="$2" tp="$3" pp="$4" cp="$5" fp8="$6" mbs="$7"
    local step="$8" mem="$9" slope="${10}"
    local k; k=$(cal_key "$wl" "$ngpu" "$tp" "$pp" "$cp" "$fp8" "$mbs")
    mkdir -p "$MLPERF_CACHE_DIR"
    if type with_lock >/dev/null 2>&1; then
        with_lock "$CAL_CACHE.lock" 30 _cal_cache_record_unlocked "$k" "$step" "$mem" "$slope"
    else
        _cal_cache_record_unlocked "$k" "$step" "$mem" "$slope"
    fi
}

cal_cache_rows_for() {
    # args: wl ngpu — emit matching rows as TSV (all TP/CP/FP8 combos).
    local wl="$1" ngpu="$2"
    [[ -f "$CAL_CACHE" ]] || return 0
    awk -F'\t' -v wl="$wl" -v ng="$ngpu" -v gm="${CAL_GPU_MODEL// /_}" '
        BEGIN{OFS="\t"}
        {
            n=split($1, p, "|")
            if (n<8) next
            if (p[1]!=gm || p[2]!=ng || p[8]!=wl) next
            # emit: tp pp cp fp8 microbs step_s mem_gb slope
            print p[3], p[4], p[5], p[6], p[7], $2, $3, $4
        }' "$CAL_CACHE"
}

# ----------------------------------------------------------------------
# Probe runner — executes a 20-step smoke via docker (or bare) for each
# candidate (tp, cp, fp8) combo, parses train_step_time, records cache.
#
# Prerequisites: $IMAGE, $DATADIR, $IMPL_DIR, $WL_NAME set by driver.
# Writes per-combo logs under $LOGDIR/calibrate/<combo>/.
# ----------------------------------------------------------------------
cal_probe() {
    local wl="$1" ngpu="$2" image="$3" datadir="$4" logdir="$5" impl="$6"
    local method="${7:-docker}"
    mkdir -p "$logdir/calibrate"

    cal_inventory_hw

    # Candidate combos: respect TP*PP*CP <= NGPU; only powers of 2 except 1.
    local -a combos=()
    local tp pp=1 cp
    for tp in 1 2 4 8; do
        (( tp > ngpu )) && continue
        for cp in 1 2 4; do
            (( tp * pp * cp > ngpu )) && continue
            (( tp * pp * cp == 0 )) && continue
            combos+=("$tp:$pp:$cp:0")   # BF16 first
        done
    done
    # FP8 variants on top combos (skip for non-Hopper/Blackwell)
    if [[ "${CAL_GPU_ARCH:-0}" -ge 89 ]]; then
        for c in "${combos[@]}"; do combos+=("${c%:0}:1"); done
    fi

    local ok=0 total=${#combos[@]}
    info "Calibration probe: $total combos, ~20 steps each (~3-5 min total)."

    local combo tp pp cp fp8 tag step_s mem slope rc
    for combo in "${combos[@]}"; do
        IFS=':' read -r tp pp cp fp8 <<<"$combo"
        tag="tp${tp}pp${pp}cp${cp}fp${fp8}"
        local cdir="$logdir/calibrate/$tag"
        mkdir -p "$cdir"
        info "  probe $tag …"

        # Build a minimal env override for this probe.
        local -a probe_env=(
            DGXNGPU="$ngpu" DGXNNODES=1
            TENSOR_MODEL_PARALLEL="$tp"
            PIPELINE_MODEL_PARALLEL="$pp"
            CONTEXT_PARALLEL="$cp"
            MICRO_BATCH_SIZE=1 MINIBS=1
            MAX_STEPS=20 VAL_CHECK_INTERVAL=9999
            WARMUP_STEPS=0 LR=0.0001
            FP8="$([[ "$fp8" == 1 ]] && echo True || echo False)"
            FP8_HYBRID="$([[ "$fp8" == 1 ]] && echo True || echo False)"
            SEQ_PARALLEL="$([[ "$tp" -gt 1 ]] && echo True || echo False)"
            TP_COMM_OVERLAP=False
            OVERLAP_PARAM_GATHER=True OVERLAP_GRAD_REDUCE=True
            USE_DIST_OPTIMIZER=True
            OVERWRITTEN_NUM_LAYERS=2
            LOGDIR="$cdir" SEED=42
        )

        rc=0
        # Build -e flags as separate argv elements — "${arr[@]/#/-e }" puts
        # the space INSIDE the quoted token so docker sees "-e KEY=VAL" as
        # a single arg and errors with "unknown flag".
        local -a eflags=()
        local _kv
        for _kv in "${probe_env[@]}"; do eflags+=(-e "$_kv"); done

        if [[ "$method" == "docker" ]] && command -v docker >/dev/null 2>&1; then
            docker run --rm --gpus "device=0-$((ngpu-1))" \
                -v "$datadir/${WL_PREPROC_HOST_SUBPATH:-}:${WL_PREPROC_MOUNT:-/preproc_data}:ro" \
                -v "$cdir:/results" \
                "${eflags[@]}" \
                "$image" bash -c "cd ${WL_CONTAINER_WORKDIR:-/workspace/llm} && \
                    bash ${WL_ENTRY:-run_and_time.sh}" \
                >"$cdir/probe.log" 2>&1 || rc=$?
        else
            (
                cd "$impl" || exit 1
                env "${probe_env[@]}" bash "${WL_ENTRY:-run_and_time.sh}"
            ) >"$cdir/probe.log" 2>&1 || rc=$?
        fi

        # Parse step-time (median of last 10 samples), peak mem, loss slope.
        step_s=$(grep -h '"train_step_time"' "$cdir"/*.log "$cdir"/*.json 2>/dev/null \
               | sed -nE 's/.*"train_step_time"[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*/\1/p' \
               | tail -10 | sort -g \
               | awk '{a[NR]=$1} END{if(NR==0) exit; if(NR%2) print a[(NR+1)/2]; else printf "%.6f\n",(a[NR/2]+a[NR/2+1])/2}')
        mem=$(grep -hE 'peak.*memory|max_memory' "$cdir"/*.log 2>/dev/null \
            | sed -nE 's/.*[^0-9]([0-9]+(\.[0-9]+)?)[[:space:]]*GB.*/\1/p' \
            | sort -g | tail -1)
        slope=$(grep -h '"reduced_train_loss"' "$cdir"/*.log "$cdir"/*.json 2>/dev/null \
              | sed -nE 's/.*"reduced_train_loss"[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*/\1/p' \
              | awk 'NR==1{first=$1} {last=$1; n=NR} END{if(n>=5) printf "%.4f\n",(last-first)/n; else print ""}')

        if [[ -n "$step_s" && "$rc" -eq 0 ]]; then
            cal_cache_record "$wl" "$ngpu" "$tp" "$pp" "$cp" "$fp8" 1 \
                "$step_s" "${mem:-0}" "${slope:-0}"
            info "    OK  step=${step_s}s  mem=${mem:-?}GB  slope=${slope:-?}"
            ok=$((ok+1))
        else
            warn "    FAIL rc=$rc  (see $cdir/probe.log)"
        fi
    done

    info "Calibration: $ok/$total combos succeeded. Cache: $CAL_CACHE"
    (( ok > 0 ))
}

# ----------------------------------------------------------------------
# Cache listing (for `--cal-list`)
# ----------------------------------------------------------------------
cal_list() {
    [[ -f "$CAL_CACHE" ]] || { info "(empty) $CAL_CACHE"; return; }
    printf '%-40s %-9s %-7s %-8s %s\n' "KEY" "STEP_S" "MEM_GB" "SLOPE" "TIMESTAMP"
    awk -F'\t' '{printf "%-40s %-9s %-7s %-8s %s\n",$1,$2,$3,$4,$5}' "$CAL_CACHE"
}
