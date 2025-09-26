#!/bin/bash

CUR_PATH=$(dirname  $(realpath $0))

if [ $# -gt 0 ]; then
    BUILD_DIR=$1
fi

[ -n "$PALLAS_PRINT_PATH" ]    || export PALLAS_PRINT_PATH=$BUILD_DIR/apps/pallas_print
[ -n "$PALLAS_INFO_PATH" ]    || export PALLAS_INFO_PATH=$BUILD_DIR/apps/pallas_info
[ -n "$PALLAS_CONFIG_PATH" ]    || export PALLAS_CONFIG_PATH=$BUILD_DIR/apps/pallas_config


C_BLACK='\033[0;30m'
C_DGRAY='\033[1;30m'
C_LGRAY='\033[0;37m'
C_WHITE='\033[1;37m'

C_GREEN='\033[0;32m'
C_LGREEN='\033[1;32m'
C_BLUE='\033[0;34m'
C_LBLUE='\033[1;34m'
C_CYAN='\033[0;36m'
C_LCYAN='\033[1;36m'
C_ORANGE='\033[0;33m'
C_YELLOW='\033[1;33m'
C_RED='\033[0;31m'
C_LRED='\033[1;31m'
C_PURPLE='\033[0;35m'
C_LPURPLE='\033[1;35m'

C_NC='\033[0m'

C_BOLD='\033[1m'


function print_error {
  echo -e "  [${C_LRED}${C_BOLD}ERROR${C_NC}] $@"
}

function print_warning {
  echo -e "[${C_YELLOW}${C_BOLD}WARNING${C_NC}] $@"
}

function print_info {
  echo -e "   [${C_LBLUE}INFO${C_NC}] ${C_BOLD}$@${C_NC}"
}

function print_ok {
  echo -e "     [${C_LGREEN}OK${C_NC}] $@"
}

function print_simple {
  echo "          $@"
}


function check_last_return {
  if [[ $? != 0 ]] ; then
    if [[ "$@" != "" ]]; then
      print_error $@
    fi
    error_occured=true
  elif [[ "$@" != "" ]] ; then
    print_ok $@
  fi
}

function check_dependencies {
    echo "> Checking for dependencies..."
    for d in $@; do
        if ! command -v $d &>/dev/null; then
            print_error "Missing command: $d"
	    ((nb_failed++))
            return 1
        fi	
    done
    ((nb_pass++))
    print_ok
    return 0
}

function check_compilation {
    echo "> Compiling test programs..."
    if [ -n "$verbose" ]; then
	make -C "$CUR_PATH"
    else
	make  -C "$CUR_PATH" > /dev/null
    fi

    if [ "$?" != "0" ]; then
	print_error "Compilation failed"
	((nb_failed++))
	return 1
    fi
    ((nb_pass++))
    print_ok
    return 0
}

function run_test {
    test=$1
    if ! [ -x "$test" ]; then
	return
    fi
    echo "> Running $test..."
    if [ -n "$verbose" ]; then
	$test
    else
	$test > /dev/null 2>&1
    fi

    if [ "$?" != "0" ]; then
        print_error "Test $test failed"
	((nb_failed++))
        return 1
    fi
    print_ok
    ((nb_pass++))
    return 0
}

function run_and_check_command {
    cmd=$@
    echo "> Running $cmd"

    ((nb_test++))
    if [ -n "$verbose" ]; then
	$cmd
    else
	$cmd > /dev/null 2>&1
    fi
    if [ "$?" != "0" ]; then
	print_error "command '$cmd' failed"
	((nb_failed++))
	return 1
    fi

    print_ok
    ((nb_pass++))
    return 0
}

function trace_get_nb_event_of_type {
    trace_filename=$1
    event_type=$2

    "$PALLAS_PRINT_PATH" "$trace_filename" 2>/dev/null |
      grep -E "^[[:space:]]+[[:digit:]]" |
      awk '{$1=""; $2=""}1' |
      grep -E -c "^[[:space:]]+$event_type"
}

function trace_get_nb_function {
    trace_filename=$1
    function_name=$2

    "$PALLAS_PRINT_PATH" "$trace_filename" 2>/dev/null |
      grep -E "^[[:space:]]+[[:digit:]]" |
      awk '{$1=""; $2=""}1' |
      grep -E -c "$function_name"
}

function trace_check_existence {
    trace_filename=$1

    ((nb_test++))
    echo " > Checking for trace existence"

    if ! [ -f  "$trace_filename" ]; then
	print_error "Cannot open trace '$trace_filename'"
	((nb_failed++))
	return 1
    else
	((nb_pass++))
	print_ok
	return 0
    fi
}

function trace_check_pallas_print {
    trace_filename=$1

    ((nb_test++))
    echo " > Checking if pallas_print works"
    echo "$PALLAS_PRINT_PATH"

    if ! "$PALLAS_PRINT_PATH" "$trace_filename"  > /dev/null 2>&1 ; then
	print_error "Cannot parse trace '$trace_filename'"
	((nb_failed++))
	return 1
    else
	((nb_pass++))
	print_ok
	return 0
    fi
}

function trace_check_pallas_info {
    trace_filename=$1

    ((nb_test++))
    echo " > Checking if pallas_info works"

    if ! "$PALLAS_INFO_PATH" "$trace_filename" > /dev/null 2>&1 ; then
	print_error "Cannot parse trace '$trace_filename'"
	((nb_failed++))
	return 1
    else
	((nb_pass++))
	print_ok
	return 0
    fi
}

function trace_check_enter_leave_parity {
    trace_filename=$1

    ((nb_test++))
    echo " > Checking for Enter/Leave parity"

    nb_enter=$(trace_get_nb_event_of_type "$trace_filename" "Enter")
    nb_leave=$(trace_get_nb_event_of_type "$trace_filename" "Leave")
    if [ $nb_enter -ne $nb_leave ]; then
	print_error "$nb_enter Enter events / $nb_leave Leave events"
	((nb_failed++))
	return 1
    else
	print_ok "$nb_enter event of each type"
	((nb_pass++))
	return 0
    fi
}


function trace_check_nb_leave {
    trace_filename=$1
    event_type="$2"
    expected_nb=$3

    ((nb_test++))
    echo " > Checking the number of Leave $event_type events"

    actual_nb=$(trace_get_nb_event_of_type "$trace_filename" "Leave")
    if [ $expected_nb -ne $actual_nb ]; then
	print_error "$actual_nb events (expected: $expected_nb)"
	((nb_failed++))
	return 1
    else
	print_ok
	((nb_pass++))
	return 0
    fi
}

function trace_check_nb_function {
    trace_filename=$1
    function_name="$2"
    expected_nb=$3

    ((nb_test++))
    echo " > Checking the number of calls to function $function_name"    

    actual_enter_nb=$(trace_get_nb_function "$trace_filename" "$function_name")
    if [ $expected_nb -ne $actual_enter_nb ]; then
	print_error "$actual_enter_nb events (expected: $expected_nb)"
	((nb_failed++))
	return 1
    else
	print_ok
	((nb_pass++))
	return 0
    fi
}


function trace_check_event_type {
    trace_filename=$1
    event_type=$2
    expected_nb=$3

    ((nb_test++))
    echo " > Checking the number of $event_type events"
    actual_nb=$(trace_get_nb_event_of_type "$trace_filename" "$event_type")

    if [ $expected_nb -ne $actual_nb ]; then
	print_error "$actual_nb events (expected: $expected_nb)"
	((nb_failed++))
	return 1
    else
	print_ok
	((nb_pass++))
	return 0
    fi
}

function trace_check_timestamp_order {
    trace_filename=$1
    thread_name=$2

    ((nb_test++))
    echo " > Checking the order of timestamps for thread $thread_name."

    timestamps=$("$PALLAS_PRINT_PATH" "$trace_filename" 2>/dev/null |grep "[[:space:]]$thread_name[[:space:]]" |awk '{print $1}')
    generated_timestamps=$(echo $timestamps | sed 's/ /\n/g')
    ordered_timestamp=$(echo $timestamps | sed 's/ /\n/g' |sort -n)
    if [ "$generated_timestamps" != "$ordered_timestamp" ]; then
	print_error "failed"
	((nb_failed++))
	return 1
    else
	print_ok
	((nb_pass++))
	return 0
    fi   
}

function trace_check_timestamp_values {
    trace_filename=$1
    thread_name=$2

    ((nb_test++))
    echo " > Checking the values of timestamps for thread $thread_name"

    timestamps=$("$PALLAS_PRINT_PATH" "$trace_filename" 2>/dev/null |grep "[[:space:]]$thread_name[[:space:]]" |awk '{print $1}' | sed 's/0\.//' | perl -pe 's/ ^0+ //xg' |tr '\n' ' ')
    nblines=$(echo $timestamps |wc -w)
    expected_timestamps=$(seq 1 $nblines |tr '\n' ' ')

    if [ "$timestamps" != "$expected_timestamps" ]; then
	print_error "failed"
	((nb_failed++))
	return 1
    else
	print_ok
	((nb_pass++))
	return 0
    fi
    
}

