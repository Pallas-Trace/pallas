#!/bin/bash

CUR_PATH=$(dirname  $(realpath $0))
source "$CUR_PATH/test_utils.sh" "$1"

BUILD_DIR=$CUR_PATH
if [ $# -gt 0 ]; then
    BUILD_DIR="$1/test"
fi

use_logical=0
n_threads=4
n_iter=40
for arg in "$@"; do
  if [ "$next_is_thread" = "1" ]; then
    n_threads="$arg";
    next_is_thread=0;
    continue
  fi
  if [ "$next_is_iter" = "1" ]; then
    n_iter="$arg";
    next_is_iter=0;
    continue
  fi
  if [ "$arg" = "-l" ]; then
    use_logical=1
  fi
  if [ "$arg" = "-n" ]; then
    next_is_iter=1;
  fi
  if [ "$arg" = "-t" ]; then
      next_is_thread=1;
  fi
done

nb_failed=0
nb_pass=0

if [ -f "$2" ]; then
  trace_dir="$(dirname -- "$(realpath -- "$2")")"
  trace_filename="$2"
else
  trace_dir="$2"
  trace_filename="$trace_dir/main.pallas"
fi
cd "$BUILD_DIR"
# trace_check_enter_leave_parity "$trace_filename"
trace_check_nb_function "$trace_filename" function_0 $(expr $n_iter \* $n_threads)
trace_check_nb_function "$trace_filename" function_1 $(expr $n_iter \* $n_threads)

if [ "$use_logical" = "1" ]; then
  for ((i=1;i<n_threads;i++)); do
      trace_check_timestamp_order "$trace_filename" thread_$i
      trace_check_timestamp_values "$trace_filename" thread_$i
  done
fi

echo "results: $nb_pass pass, $nb_failed failed"
if [ $nb_failed -gt 0 ]; then
  exit 1;
else
  rm -rf "$trace_dir"
  exit 0;
fi
