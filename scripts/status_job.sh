#!/bin/bash
echo  `pwd`
glite-wms-job-status --noint --output tmpstatus $1
status=$?
if (( status==0 )); then
  nlines=$(cat tmpstatus | wc -l)
  total=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'| wc -l)
  scheduled=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Scheduled | wc -l)
  running=$(tail -$((nlines - 10)) tmpstatus  | grep 'Current Status:'[[:blank:]]*Running | wc -l)
  canceled=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Cancelled | wc -l)
  aborted=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Aborted | wc -l)
  cleared=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Cleared | wc -l)
  purged=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Purged | wc -l)
  waiting=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Waiting | wc -l)
  ready=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Ready | wc -l)
  submitted=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Submitted | wc -l)
  donejobs=$(tail -$((nlines - 10)) tmpstatus  | grep 'Current Status:'[[:blank:]]*Done | wc -l)
  successful_jobs=$(tail -$((nlines - 10)) tmpstatus  |  grep 'Current Status:'[[:blank:]]*Done\(Success\) | wc -l)
  unsucessful_jobs=$(tail -$((nlines - 10)) tmpstatus  |  grep 'Current Status:'[[:blank:]]*Done\('Exit Code !=0'\) | wc -l)
  echo "Total jobs : $total"
  if (( submitted>0 )); then
    echo "Submitted : $submitted"
  fi
  if (( ready>0 )); then
    echo "Ready : $ready"
  fi

  if (( waiting>0 )); then
    echo "Waiting : $waiting"
  fi

  if (( scheduled>0 )); then
    echo "Scheduled : $scheduled"
  fi

  if (( running>0 )); then
    echo "Running : $running"
  fi
  if (( donejobs>0 )); then
    echo "Done : $donejobs"
    echo "|"
    echo "+--- Sucess : $successful_jobs"
    echo "+----Failed : $unsucessful_jobs"
  fi
  if (( canceled>0 )); then
    echo "Cancelled : $canceled"
  fi
  if (( aborted>0 )); then
    echo "Aborted : $aborted"
  fi
  if (( cleared>0 )); then
    echo "Cleared : $cleared"
  fi
  if (( purged>0 )); then
    echo "Purged : $purged"
  fi


else
  echo "Cannot retrieve information for task  $1"
fi
#rm tmpstatus

