#!/bin/bash

CUR_PATH=$(dirname  $(realpath $0))
source "$CUR_PATH/test_utils.sh" "$1"



if which wget &> /dev/null; then
  DOWNLOADER="wget"
else
  print_error "wget not found"
  exit 0
fi
print_info "Downloader is $DOWNLOADER"
if ! which tar &> /dev/null; then
  print_error "Tar not found on the system"
  exit 0
fi


PALLAS_ABI=$("$PALLAS_CONFIG_PATH" | grep ABI -m 1 | cut -d ' ' -f 3)
print_info "Pallas ABI: ${PALLAS_ABI}"
mkdir -p "ABI_$PALLAS_ABI"
cd "ABI_$PALLAS_ABI" || print_error "Folder ABI_${PALLAS_ABI} wasn't created"
URI="http://stark.int-evry.fr/traces/pallas_traces/ABI_${PALLAS_ABI}"


function download() {
  ARCHIVE_PATH="$1.tgz"
  URL="${URI}/${ARCHIVE_PATH}"
  print_info "Downloading $URL"
#  if [ -e "$ARCHIVE_PATH" ]; then
#    print_ok "Trace already downloaded."
  if ${DOWNLOADER} -O "${ARCHIVE_PATH}" "${URL}" &> /dev/null; then
    print_ok "Trace downloaded."
  else
    return 1;
  fi
  if tar -xzvf "${ARCHIVE_PATH}" &> /dev/null; then
    print_ok "Trace decompressed."
    return 0;
  else
    return 1;
  fi
}

function test() {
  print_info "Testing $1"
  TRACE_PATH="$1_trace"
  if ! download "$TRACE_PATH"; then
    print_error "Could not download the trace"
    exit 1
  fi
  start=`date +%s`


  COMMAND="$PALLAS_INFO_PATH -t -D -la -lt --content -da $TRACE_PATH/eztrace_log.pallas"
  print_info "$COMMAND"
  if ! eval "$COMMAND"  &> /dev/null; then
    print_error "pallas_info $1 failed."
    return 1
  else
    end=`date +%s`
    print_ok "pallas_info $1 in $((end-start))s."
  fi


  COMMAND="$PALLAS_PRINT_PATH -T $TRACE_PATH/eztrace_log.pallas"
  print_info "$COMMAND"
  start=`date +%s`
  if ! eval "$COMMAND"  &> /dev/null; then
        print_error "pallas_print -T $1 failed."
        return 1
  else
      end=`date +%s`
      print_ok "pallas_print -T $1 in $((end-start))s."
  fi


  COMMAND="$PALLAS_PRINT_PATH -S $TRACE_PATH/eztrace_log.pallas"
  print_info "$COMMAND"
  start=`date +%s`
  if ! eval "$COMMAND"  &> /dev/null; then
        print_error "pallas_print -S $1 failed."
        return 1
  else
      end=`date +%s`
      print_ok "pallas_print -S $1 in $((end-start))s."
  fi


  COMMAND="$PALLAS_PRINT_PATH $TRACE_PATH/eztrace_log.pallas"
  print_info "$COMMAND"
  start=`date +%s`
  if ! eval "$COMMAND"  &> /dev/null; then
      print_error "pallas_print $1 failed."
      return 1
  else
      end=`date +%s`
      print_ok "pallas_print $1 in $((end-start))s."
  fi
}

test "$2"