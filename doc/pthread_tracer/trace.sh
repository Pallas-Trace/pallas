#!/bin/bash

prefix=$(dirname $(realpath $0))
libdir="$prefix"

debug=n
verbose=

usage()
{
cat << EOF
usage: $0 options

OPTIONS:
   -?                               Show this message
   -d                               Run with gdb
   -v                               Verbose mode
   -V                               Ultra verbose mode
EOF
}

while getopts 'dvV' OPTION; do
  case $OPTION in
  d)
	debug=y
       	;;
  v)
	verbose=verbose
       	;;
  V)
        verbose=max
       	;;
  ?)	usage
	exit 2
	;;
  esac
done

# remove the options from the command line
shift $(($OPTIND - 1))


LD_PRELOAD="$libdir/libpthread_tracer.so"

if [ x$debug = xy ]; then

#  generate a gdbinit file that will preload all the modules

   gdbinit_file=`mktemp`
   echo "set env @LD_PRELOAD_NAME@ $LD_PRELOAD" > $gdbinit_file
   if [ -n "$verbose" ]; then
       echo "set env VERBOSE $verbose" > $gdbinit_file
   fi
   echo "echo \n" >> $gdbinit_file
   echo "echo trace.sh: hook loaded\n" >> $gdbinit_file

   gdb -x $gdbinit_file  --args $@
   rm $gdbinit_file

else
    if [ -n "$verbose" ]; then
	export VERBOSE=$verbose
    fi
    @LD_PRELOAD_NAME@=$LD_PRELOAD $@
fi   
