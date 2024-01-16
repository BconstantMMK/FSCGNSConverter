#!/bin/bash

NProc=0
NThd=0
CMD=""
CurrentArg=""

help()
{
  echo
  echo "Usage : run_MPI-OPENMP -n <nbProcesses> -t <nbThreads> -cmd <command>"
  echo
  echo "Description : Run Hybrid MPI-OPENMP application with smart process/thread distribution and binding"
  echo
  echo "-n <nbProcesses> : Number of MPI processes to execute"
  echo
  echo "-t <nbThreads> : Number of OPENMP Threads by MPI process to execute"
  echo 
  echo "-cmd <command> : Command to execute in the MPI-OPENMP context"
  echo "                 (White spaces are handled)"
}

while [ "$1" != "" ]; do
  CurrentArg=$1
  case $1 in
    -n | --nbProcesses ) shift
                         NProc=$1
                         ;;
    -t | --nbThreads ) shift
                       NThd=$1
                       ;;
    -cmd | --command ) shift
                       while [[ "$1" != "" && ${1:0:1} != "-" ]]; do
                       CMD=$CMD" "$1
                       shift
                       done
                       ;;
    -h | --help ) help
                  exit 1
                  ;;
     * ) echo "Unknow command argument \""$CurrentArg"\""
         echo "use --help for help display"
         exit 1
    esac
    shift
    if [[ "$1" != "" && ${1:0:1} != "-" ]]
    then
      echo "Unexpected value \""$1"\" in argument "\"$CurrentArg"\""
      exit 1
    fi
done

if [[ "$NProc" -eq 0 ]]
then
echo "Missing option -n <nbProcesses>"
exit 1
fi

if [[ "$NThd" -eq 0 ]]
then
echo "Missing option -t <nbThreads>"
exit 1
fi

if [ "$CMD" == "" ]
then
echo "Missing option -cdm <command>"
exit 1
fi

CMD2="mpirun -np "$NProc" --bind-to core --map-by socket:PE="$NThd" --report-bindings -x OMP_NUM_THREADS="$NThd" -x OMP_PLACES=cores -x OMP_PROC_BIND=TRUE -x OMP_DISPLAY_ENV=VERBOSE "$CMD

echo $CMD2

$CMD2
