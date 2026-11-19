#!/bin/bash
# [PF] end to end for one CLD sample: the [T] tracks fitted with Genfit2 (steps A-C of
# ../tracking/genfit/run_chain.sh), the [PF] shard cache (data.py, which also runs step D),
# and the training (train.py). Every stage skips what exists, so the script can be rerun after
# an interruption or run stage by stage: the chain and the cache need CPUs only, the training a
# GPU.
#
#   run_pf.sh -s SOURCES -w WORK -m NAME=CKPT -o OUTDIR [-n NFILES] [-j JOBS] [-a NA -t TA] \
#       [-S chain,cache,train] [-- <train.py options>]
#
#   -s SOURCES  EDM4hep REC source files of the sample, one per line
#   -w WORK     work directory: sources.txt, chain/ (run_chain.sh), cache/ (shards)
#   -m N=CKPT   [T] model name and checkpoint (one model)
#   -o OUTDIR   training run directory
#   -n NFILES   use the first NFILES of the sorted sources (default all); the held-out set is
#               the first 10% of them
#   -j JOBS     parallel Genfit fits and shard builders (default 8)
#   -a NA       run step A ([T] inference) in NA parallel processes of -t threads each
#               (default 1: run_chain.sh's single process; use NA > 1 when step A runs on CPU)
#   -g NGPU     train on NGPU GPUs (torchrun, data parallel; default 1)
#   -S STAGES   comma-separated subset of chain,cache,train (default all three)
#
# Environment: TORCH_PY = python with torch/uproot/pyarrow (default python3); KEY4HEP_SETUP as
# in run_chain.sh.
set -eo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
J=8; N=0; NA=1; TA=4; NG=1; STAGES=chain,cache,train
while getopts "s:w:m:o:n:j:a:t:g:S:" o; do
  case $o in
    s) SRC=$OPTARG ;; w) W=$OPTARG ;; m) M=$OPTARG ;; o) OUT=$OPTARG ;;
    n) N=$OPTARG ;; j) J=$OPTARG ;; a) NA=$OPTARG ;; t) TA=$OPTARG ;; g) NG=$OPTARG ;; S) STAGES=$OPTARG ;;
    *) sed -n 2,26p "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
[ "$1" = "--" ] && shift
[ -f "$SRC" ] && [ -n "$W" ] && [ -n "$M" ] || { sed -n 2,26p "$0"; exit 1; }
export TORCH_PY=${TORCH_PY:-python3}
NAME=${M%%=*}
mkdir -p "$W"
W=$(cd "$W" && pwd)
LIST=$W/sources.txt
sort "$SRC" | awk -v n="$N" 'n == 0 || NR <= n' > "$LIST"     # awk, not head: no SIGPIPE under pipefail
echo "$(date +%T) $(wc -l < "$LIST") source files, model $NAME, stages $STAGES"

if [[ ",$STAGES," == *",chain,"* ]]; then
  if [ "$NA" -gt 1 ]; then
    # step A in NA parallel processes (it skips files already labelled, so run_chain.sh's own
    # single-process step A then finds every file done); worth it when step A runs on CPU
    mkdir -p "$W/chain/labels_$NAME" "$W/chain/logs"
    split -n "l/$NA" -d "$LIST" "$W/chain/stepA_part_"
    for p in "$W"/chain/stepA_part_*; do
      OMP_NUM_THREADS=$TA "$TORCH_PY" -u "$H/../tracking/genfit/export_infer.py" --detector cld \
          --ckpt "${M#*=}" --files $(cat "$p") --outdir "$W/chain/labels_$NAME" \
          >> "$W/chain/logs/A_${NAME}_$(basename "$p").log" 2>&1 &
    done
    wait
    rm -f "$W"/chain/stepA_part_*
    echo "$(date +%T) parallel step A done ($NA processes)"
  fi
  bash "$H/../tracking/genfit/run_chain.sh" -d cld -w "$W/chain" -f "$LIST" -m "$M" -j "$J"
fi
cd "$H"
if [[ ",$STAGES," == *",cache,"* ]]; then
  "$TORCH_PY" -u data.py --sources "$LIST" --chain-work "$W/chain" --model "$NAME" \
      --shard-cache "$W/cache" --jobs "$J" --test-frac 0.1
fi
if [[ ",$STAGES," == *",train,"* ]]; then
  [ -n "$OUT" ] || { echo "-o OUTDIR is required for the train stage"; exit 1; }
  if [ "$NG" -gt 1 ]; then RUN=("$TORCH_PY" -m torch.distributed.run --standalone --nproc_per_node "$NG");
  else RUN=("$TORCH_PY" -u); fi
  "${RUN[@]}" train.py --sources "$LIST" --chain-work "$W/chain" --model "$NAME" \
      --shard-cache "$W/cache" --outdir "$OUT" "$@"
fi
