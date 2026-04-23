# shellcheck shell=bash
# Dynamic recipe derivation from calibration cache.
# Sourced by mlperf.sh. Depends on common.sh + calibrate.sh.

[[ -n "${_MLPERF_RECIPES_LOADED:-}" ]] && return 0
_MLPERF_RECIPES_LOADED=1

# Reference GBS per workload (used for LR scaling anchor).
declare -A _REC_REF_GBS=(
    [llama31_8b]=1152
    [llama31_405b]=1536
    [llama2_70b_lora]=8
    [flux1]=256
    [retinanet]=128
    [rgat]=1024
    [dlrm_dcnv2]=65536
)
# Reference peak LR (matches config_common_*.sh).
declare -A _REC_REF_LR=(
    [llama31_8b]=0.0008
    [llama31_405b]=0.00015
    [llama2_70b_lora]=0.0004
    [flux1]=0.0001
    [retinanet]=0.0001
    [rgat]=0.001
    [dlrm_dcnv2]=0.005
)
# Reference total steps to convergence target.
declare -A _REC_REF_STEPS=(
    [llama31_8b]=1200000
    [llama31_405b]=6500
    [llama2_70b_lora]=1000
    [flux1]=80000
    [retinanet]=30
    [rgat]=1500
    [dlrm_dcnv2]=75000
)
# Per-eval wallclock estimate (seconds).
declare -A _REC_EVAL_S=(
    [llama31_8b]=70 [llama31_405b]=600 [llama2_70b_lora]=40
    [flux1]=30 [retinanet]=45 [rgat]=20 [dlrm_dcnv2]=15
)

# sqrt(GBS/ref_GBS) * ref_LR — stable LR heuristic for small-batch regime.
rec_derive_lr() {
    local wl="$1" gbs="$2"
    local ref_gbs="${_REC_REF_GBS[$wl]:-1024}"
    local ref_lr="${_REC_REF_LR[$wl]:-0.0002}"
    awk -v g="$gbs" -v rg="$ref_gbs" -v rl="$ref_lr" \
        'BEGIN{ if (g<=0||rg<=0) print rl; else printf "%.6f\n", rl * sqrt(g/rg) }'
}

# max(2000, 0.01 * max_steps), but never > 10% of max_steps.
rec_derive_warmup() {
    local max_steps="$1"
    awk -v s="$max_steps" 'BEGIN{
        if (s+0 <= 0) { print 0; exit }
        w1 = 2000; w2 = int(s*0.01); w=(w1>w2?w1:w2)
        hi = int(s*0.10); if (w>hi) w=hi
        if (w<10) w=10
        print w
    }'
}

# VCI so that n_evals * eval_s stays <20% of train time.
rec_derive_vci() {
    local max_steps="$1" step_s="$2" eval_s="$3"
    awk -v ms="$max_steps" -v sp="$step_s" -v ev="$eval_s" 'BEGIN{
        train_s = ms*sp
        budget = train_s * 0.20
        max_evals = (ev>0)? int(budget/ev) : 10
        if (max_evals<1) max_evals=1
        vci = int(ms/max_evals)
        if (vci<100) vci=100
        print vci
    }'
}

# rec_eta wl step_s max_steps vci -> seconds
rec_eta() {
    local wl="$1" step_s="$2" max_steps="$3" vci="$4"
    local eval_s="${_REC_EVAL_S[$wl]:-60}"
    awk -v sp="$step_s" -v ms="$max_steps" -v vci="$vci" -v ev="$eval_s" 'BEGIN{
        train_s = ms*sp
        ne = (vci>0)? int(ms/vci) : 0
        printf "%d\n", train_s + ne*ev
    }'
}

_rec_fmt_dur() {
    local s="${1:-0}"; s=${s%.*}
    (( s <= 0 )) && { echo "0m"; return; }
    local h=$((s/3600)) m=$(((s%3600)/60))
    (( h > 24 )) && { printf '%dd%dh\n' $((h/24)) $((h%24)); return; }
    (( h > 0 )) && printf '%dh%02dm\n' "$h" "$m" || printf '%dm\n' "$m"
}

# ----------------------------------------------------------------------
# Find the best measured step-time for a (wl, ngpu) tuple across the
# calibration cache. Ranks by BF16 first, then lowest step_s.
# Emits: "tp pp cp fp8 step_s mem_gb slope" or empty.
# ----------------------------------------------------------------------
rec_best_probe() {
    local wl="$1" ngpu="$2" prefer_fp8="${3:-0}"
    local rows; rows=$(cal_cache_rows_for "$wl" "$ngpu")
    [[ -z "$rows" ]] && return 1
    # Output columns (space-separated): tp pp cp fp8 microbs step_s mem_gb slope
    awk -F'\t' -v pf="$prefer_fp8" '
        { rows[NR]=$0; fp8[NR]=$4; step[NR]=$6 }
        END{
            best=0; bv=1e18
            for (i=1;i<=NR;i++) if (fp8[i]==pf && step[i]+0<bv){bv=step[i]+0; best=i}
            if (best==0) for (i=1;i<=NR;i++) if (step[i]+0<bv){bv=step[i]+0; best=i}
            if (best>0) { split(rows[best],a,"\t")
                printf "%s %s %s %s %s %s %s %s\n", a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8] }
        }' <<<"$rows"
}

# ----------------------------------------------------------------------
# Build and display a dynamic recipe menu for (wl, ngpu).
# Requires cal cache populated. Sets REC_CHOICE_* globals on select.
# ----------------------------------------------------------------------
rec_menu() {
    local wl="$1" ngpu="$2"
    local ref_gbs="${_REC_REF_GBS[$wl]:-1024}"
    local ref_steps="${_REC_REF_STEPS[$wl]:-100000}"
    local eval_s="${_REC_EVAL_S[$wl]:-60}"

    # Best measured combo (BF16).
    local bf16_line fp8_line
    bf16_line=$(rec_best_probe "$wl" "$ngpu" 0) || true
    fp8_line=$(rec_best_probe "$wl" "$ngpu" 1) || true

    if [[ -z "$bf16_line" ]]; then
        warn "No calibration data for $wl at ${ngpu} GPUs. Run with --calibrate first."
        return 1
    fi

    local b_tp b_pp b_cp b_fp8 b_mbs b_step
    read -r b_tp b_pp b_cp b_fp8 b_mbs b_step _rest <<<"$bf16_line"
    local f_tp f_pp f_cp f_fp8 f_mbs f_step
    if [[ -n "$fp8_line" ]]; then
        read -r f_tp f_pp f_cp f_fp8 f_mbs f_step _rest <<<"$fp8_line"
    fi
    local dp=$(( ngpu / (b_tp * b_pp * b_cp) ))
    (( dp < 1 )) && dp=1

    say "Dynamic recipes (based on measured step-time on this hardware)"

    # Build 5 recipes: Smoke, Shape, Short conv, Full conv, FP8 throughput.
    local -a R_NAME R_DESC R_STEPS R_GBS R_LR R_WARM R_VCI R_ETA R_FP8
    local -a R_TP R_PP R_CP

    _add() {
        R_NAME+=("$1"); R_DESC+=("$2"); R_STEPS+=("$3"); R_GBS+=("$4")
        R_LR+=("$5"); R_WARM+=("$6"); R_VCI+=("$7"); R_ETA+=("$8")
        R_FP8+=("$9"); R_TP+=("${10}"); R_PP+=("${11}"); R_CP+=("${12}")
    }

    # 1. Smoke — forces TP=PP=CP=1, so actual DP=ngpu, GBS=ngpu.
    _add "Smoke"          "quick image/data/NCCL check"         20 "$ngpu" "0.0001" 0   9999 \
         "$(rec_eta "$wl" "$b_step" 20 9999)" 0 1 1 1

    # 2. Shape
    local shape_steps=500
    _add "Shape check"    "throughput benchmark, no convergence" \
         "$shape_steps" "$dp" "0.0001" 50 9999 \
         "$(rec_eta "$wl" "$b_step" "$shape_steps" 9999)" 0 "$b_tp" "$b_pp" "$b_cp"

    # 3. Short convergence — GBS grown via MINIBS to stabilize LR.
    local sc_minibs=64
    local sc_gbs=$(( sc_minibs * dp ))
    (( sc_gbs < 32 )) && sc_gbs=32
    local sc_steps=50000
    local sc_lr sc_warm sc_vci
    sc_lr=$(rec_derive_lr "$wl" "$sc_gbs")
    sc_warm=$(rec_derive_warmup "$sc_steps")
    sc_vci=$(rec_derive_vci "$sc_steps" "$b_step" "$eval_s")
    local sc_eff_step; sc_eff_step=$(awk -v s="$b_step" -v m="$sc_minibs" 'BEGIN{printf "%.4f", s*m}')
    _add "Short convergence"  "GBS=${sc_gbs}, loss<4 target" \
         "$sc_steps" "$sc_gbs" "$sc_lr" "$sc_warm" "$sc_vci" \
         "$(rec_eta "$wl" "$sc_eff_step" "$sc_steps" "$sc_vci")" 0 "$b_tp" "$b_pp" "$b_cp"

    # 4. Full convergence — reference GBS via MINIBS, full steps.
    local fc_minibs=$(( ref_gbs / dp ))
    (( fc_minibs < 1 )) && fc_minibs=1
    local fc_gbs=$(( fc_minibs * dp ))
    local fc_lr fc_warm fc_vci
    fc_lr="${_REC_REF_LR[$wl]:-0.0002}"
    fc_warm=$(rec_derive_warmup "$ref_steps")
    fc_vci=$(rec_derive_vci "$ref_steps" "$b_step" "$eval_s")
    local fc_eff_step; fc_eff_step=$(awk -v s="$b_step" -v m="$fc_minibs" 'BEGIN{printf "%.4f", s*m}')
    _add "Full convergence"   "reference GBS=${fc_gbs}, MLPerf target" \
         "$ref_steps" "$fc_gbs" "$fc_lr" "$fc_warm" "$fc_vci" \
         "$(rec_eta "$wl" "$fc_eff_step" "$ref_steps" "$fc_vci")" 0 "$b_tp" "$b_pp" "$b_cp"

    # 5. FP8 throughput (if measured)
    if [[ -n "$fp8_line" ]]; then
        local fp_eff_step; fp_eff_step=$(awk -v s="$f_step" -v m="$sc_minibs" 'BEGIN{printf "%.4f", s*m}')
        _add "FP8 throughput"     "same as short-conv but FP8 (~${f_step}s/step)" \
             "$sc_steps" "$sc_gbs" "$sc_lr" "$sc_warm" "$sc_vci" \
             "$(rec_eta "$wl" "$fp_eff_step" "$sc_steps" "$sc_vci")" 1 "$f_tp" "$f_pp" "$f_cp"
    fi

    local n=${#R_NAME[@]} i
    for ((i=0; i<n; i++)); do
        printf "  [%d] %-20s  steps=%-8s  GBS=%-6s  LR=%-10s  ETA=%s  (%s)\n" \
            "$((i+1))" "${R_NAME[i]}" "${R_STEPS[i]}" "${R_GBS[i]}" \
            "${R_LR[i]}" "$(_rec_fmt_dur "${R_ETA[i]}")" "${R_DESC[i]}"
    done
    printf "  [%d] %-20s  (hand-tune all knobs)\n" "$((n+1))" "Custom"

    local choice
    if (( MLPERF_AUTO_YES == 1 )); then choice=1
    else
        while :; do
            read -r -p "Pick recipe [1]: " choice
            choice="${choice:-1}"
            [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= n+1 )) && break
            err "Enter 1..$((n+1))"
        done
    fi

    if (( choice == n+1 )); then
        REC_CHOICE_NAME="custom"
        return 2   # signal: fall through to legacy config-file flow
    fi
    local idx=$((choice-1))
    REC_CHOICE_NAME="${R_NAME[idx]}"
    REC_CHOICE_STEPS="${R_STEPS[idx]}"
    REC_CHOICE_GBS="${R_GBS[idx]}"
    REC_CHOICE_LR="${R_LR[idx]}"
    REC_CHOICE_WARMUP="${R_WARM[idx]}"
    REC_CHOICE_VCI="${R_VCI[idx]}"
    REC_CHOICE_ETA_S="${R_ETA[idx]}"
    REC_CHOICE_FP8="${R_FP8[idx]}"
    REC_CHOICE_TP="${R_TP[idx]}"
    REC_CHOICE_PP="${R_PP[idx]}"
    REC_CHOICE_CP="${R_CP[idx]}"
    # Guard: if the selected recipe's parallelism differs from the best-BF16
    # probe (e.g. FP8 row had different TP), dp can hit zero via integer
    # truncation. Clamp before the divide.
    local _rec_dp=$(( ngpu / (REC_CHOICE_TP * REC_CHOICE_PP * REC_CHOICE_CP) ))
    (( _rec_dp < 1 )) && _rec_dp=1
    REC_CHOICE_MINIBS=$(( REC_CHOICE_GBS / _rec_dp ))
    (( REC_CHOICE_MINIBS < 1 )) && REC_CHOICE_MINIBS=1
    info "Selected: $REC_CHOICE_NAME  →  ETA $(_rec_fmt_dur "$REC_CHOICE_ETA_S")"
    return 0
}

# Export chosen recipe into the env the launcher will see. Call right
# before the launch dispatch.
rec_export() {
    [[ -z "${REC_CHOICE_NAME:-}" || "$REC_CHOICE_NAME" == "custom" ]] && return 0
    export MAX_STEPS="$REC_CHOICE_STEPS"
    export VAL_CHECK_INTERVAL="$REC_CHOICE_VCI"
    export WARMUP_STEPS="$REC_CHOICE_WARMUP"
    export LR="$REC_CHOICE_LR"
    export TENSOR_MODEL_PARALLEL="$REC_CHOICE_TP"
    export PIPELINE_MODEL_PARALLEL="$REC_CHOICE_PP"
    export CONTEXT_PARALLEL="$REC_CHOICE_CP"
    export MICRO_BATCH_SIZE=1
    export MINIBS="$REC_CHOICE_MINIBS"
    if [[ "$REC_CHOICE_FP8" == "1" ]]; then
        export FP8=True FP8_HYBRID=True
    else
        export FP8=False FP8_HYBRID=False
    fi
    export GRADIENT_CLIP_VAL=1.0
    [[ "$REC_CHOICE_TP" -gt 1 ]] && export SEQ_PARALLEL=True || export SEQ_PARALLEL=False
    info "Exported recipe env: MAX_STEPS=$MAX_STEPS GBS=$REC_CHOICE_GBS LR=$LR WARMUP=$WARMUP_STEPS VCI=$VAL_CHECK_INTERVAL TP=$TENSOR_MODEL_PARALLEL CP=$CONTEXT_PARALLEL FP8=$FP8"
}
