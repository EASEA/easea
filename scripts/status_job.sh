#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage status_job  jobid"
  exit 1
fi

glite-wms-job-status --noint --output tmpstatus $1 >& /dev/null
status=$?

if (( status==0 )); then
  nlines=$(cat tmpstatus | wc -l)
  global_status=$(head tmpstatus | sed -n 's/Current Status:\ *\(.*\).*/\1/p')
  scheduled=0
  running=0
  cancelled=0
  aborted=0
  purged=0
  waiting=0
  ready=0
  sucessful_jobs=0
  unsucessful_jobs=0
  total=0
  old_IFS=$IFS     # sauvegarde du séparateur de champ  
  IFS=$'\n' 
  for worker_status in $(tail -$((nlines - 10)) tmpstatus | sed -n 's/.\+Current Status:\ *\(.*\).*/\1/p' )
  do
     case $worker_status in
        Scheduled)
            scheduled=$((scheduled+1))
            ;;
        Running)
            running=$((running+1))
            ;;
        Cancelled)
            cancelled=$((cancelled+1))
	    ;;
        Aborted)
            aborted=$((aborted+1))
            ;;
        Cleared)
            cleared=$((cleared+1))
	    ;;
        Purged)
            purged=$((purged+1))
	    ;;
        Waiting)
            waiting=$((waiting+1))
	    ;;
        Ready)
            ready=$((ready+1)) 
	    ;;
        Submitted)
            submitted=$((submitted+1)) 
	    ;;
        Done\(Success\))
            successful_jobs=$((successful_jobs+1))
	    ;;
        Done\(Exit\ Code\ !=0\))
            unsucessful_jobs=$((unsucessful_jobs+1)) 
	    ;;
     esac
     donejobs=$((successful_jobs+unsucessful_jobs))
     total=$((total+1))
  done  
  IFS=$old_IFS  	

#fi


#if (( status==0 )); then
#   nlines=$(cat tmpstatus | wc -l)
#   global_status=$(head tmpstatus | sed -n 's/Current Status:[[:blank:]]*\(.*\).*/\1/p')
#   total=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'| wc -l)
#   scheduled=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Scheduled | wc -l)
#   running=$(tail -$((nlines - 10)) tmpstatus  | grep 'Current Status:'[[:blank:]]*Running | wc -l)
#   canceled=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Cancelled | wc -l)
#   aborted=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Aborted | wc -l)
#   cleared=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Cleared | wc -l)
#   purged=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Purged | wc -l)
#   waiting=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Waiting | wc -l)
#   ready=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Ready | wc -l)
#   submitted=$(tail -$((nlines - 10)) tmpstatus | grep 'Current Status:'[[:blank:]]*Submitted | wc -l)
#   donejobs=$(tail -$((nlines - 10)) tmpstatus  | grep 'Current Status:'[[:blank:]]*Done | wc -l)
#   successful_jobs=$(tail -$((nlines - 10)) tmpstatus  |  grep 'Current Status:'[[:blank:]]*Done\(Success\) | wc -l)
#   unsucessful_jobs=$(tail -$((nlines - 10)) tmpstatus  |  grep 'Current Status:'[[:blank:]]*Done\('Exit Code !=0'\) | wc -l)
  echo "Global status: $global_status"
  echo "Total jobs   : $total"
  echo "|"
  if (( submitted>0 )); then
    echo "+--Submitted : $submitted"
  fi
  if (( ready>0 )); then
    echo "+--Ready : $ready"
  fi

  if (( waiting>0 )); then
    echo "+--Waiting : $waiting"
  fi

  if (( scheduled>0 )); then
    echo "+--Scheduled : $scheduled"
  fi

  if (( running>0 )); then
    echo "+--Running : $running"
  fi
  if (( donejobs>0 )); then
    echo "+--Done : $donejobs"
    echo "   |"
    echo "   +--- Sucess : $successful_jobs"
    echo "   +----Failed : $unsucessful_jobs"
  fi
  if (( cancelled>0 )); then
    echo "+--Cancelled : $cancelled"
  fi
  if (( aborted>0 )); then
    echo "+--Aborted : $aborted"
  fi
  if (( cleared>0 )); then
    echo "Cleared : $cleared"
  fi
  if (( purged>0 )); then
    echo "+--Purged : $purged"
  fi


else
  echo "Cannot retrieve information for task  $1"
fi
#rm tmpstatus

