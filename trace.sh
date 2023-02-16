#!/bin/bash

prefix=$(dirname $(realpath $0))

debug=n
usage()
{
cat << EOF
usage: $0 options

OPTIONS:
   -?                               Show this message
   -d                               Run with gdb
EOF
}

while getopts 'd' OPTION; do
  case $OPTION in
  d)
	debug=y
       	;;
  ?)	usage
	exit 2
	;;
  esac
done

# remove the options from the command line
shift $(($OPTIND - 1))


LD_PRELOAD="$prefix/liblock.so"

if [ x$debug = xy ]; then

#  generate a gdbinit file that will preload all the modules

   gdbinit_file=`mktemp`
   echo "set env LD_PRELOAD $LD_PRELOAD" > $gdbinit_file
   echo "echo \n" >> $gdbinit_file
   echo "echo trace.sh: hook loaded\n" >> $gdbinit_file

   gdb -x $gdbinit_file  --args $@
   rm $gdbinit_file

else
    LD_PRELOAD=$LD_PRELOAD $@
fi   
