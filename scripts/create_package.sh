#!/bin/bash
EXECUTABLE_NAME=$1
EXECUTABLE_PATH=$2
PACKAGE_DESTINATION=$3
NUM_MACHINES=$4
TMPDIR="tmp_package"


mkdir $TMPDIR >& /dev/null

if [ -z  "$EZ_PATH" ]; then
  echo "Variable EZ_PATH not found, please set the EZ_PATH variable"
  exit 1
fi


if [ -z "$1" ]; then
   echo "Usage executable-name [executable-path] [package_destination]"
   exit 1
fi

if [ -z "$2" ]; then
   EXECUTABLE_PATH=$EXECUTABLE_NAME;
   PACKAGE_DESTINATION=`pwd`  
   echo "Assuming executable path = $EXECUTABLE_PATH"
   echo "Package destination = $PACKAGE_DESTINATION"
fi

if [ ! -d "$EXECUTABLE_PATH" ]; then
   echo "Executable directory not found ... exiting"
   exit 1
fi

EXECUTABLE_FULLNAME="${EXECUTABLE_PATH}/${EXECUTABLE_NAME}"

if [ ! -x "$EXECUTABLE_FULLNAME" ]; then
    echo "Executable file not found"
    exit 1
fi


EXECUTABLE_PARAMFILENAME="${EXECUTABLE_NAME}.prm"
EXECUTABLE_FULLPARMFILENAME="${EXECUTABLE_FULLNAME}.prm"

if [ ! -f "$EXECUTABLE_FULLPARMFILENAME" ]; then
    echo "Parameter file not found"
    exit 1
fi

CERTIFICATE_FILE=$X509_USER_PROXY


voms-proxy-info -exists >& /dev/null

if [ $? -eq 1 ]; then
   echo "No certificate found, please create one"
   exit 1
fi

voms-proxy-info -exists -valid 1:0 >& /dev/null

if [ $? -eq 1 ]; then
   echo "Your certificate will expire soon (less than 1 hour), please renew it"
   exit 1
fi

#TIMELEFT=`voms-proxy-info -timeleft`


if  [ ! -f "${EZ_PATH}/scripts/cde_template.jdl"  ] ||  [ ! -f "${EZ_PATH}/scripts/execute-cde_template.sh" ] ; then
  echo "Template files not found, please check you have correctly the EZ_PATH variable"
  exit 1
fi


cp $EXECUTABLE_FULLNAME "${TMPDIR}/."
cp $EXECUTABLE_FULLPARMFILENAME "${TMPDIR}/."
cp $CERTIFICATE_FILE "${TMPDIR}/certificate"

PACKAGE_FILENAME="${PACKAGE_DESTINATION}/application.tar.gz"

echo "Creating file $PACKAGE_FILENAME"

tar -C $TMPDIR -cvzf $PACKAGE_FILENAME $EXECUTABLE_NAME $EXECUTABLE_PARAMFILENAME certificate  >& /dev/null

status=$?
if (( status==0 )); then
  echo "Package created ... sucess!"
else
  echo "Package creation failed !!!"
  exit 1
fi

cp "${EZ_PATH}/scripts/cde_template.jdl" "${PACKAGE_DESTINATION}/application.jdl" >& /dev/null
cp "${EZ_PATH}/scripts/execute-cde_template.sh" "${PACKAGE_DESTINATION}/execute-cde.sh" >& /dev/null

if [ -z "$4" ]; then
   NUM_MACHINES="50";
   echo "Number of machines not specified, default to $NUM_MACHINES"
fi

sed -i "s/APPLICATIONNAME/${EXECUTABLE_NAME}/g" "${PACKAGE_DESTINATION}/execute-cde.sh" >& /dev/null
sed -i "s/NUMBEROFMACHINES/${NUM_MACHINES}/g" "${PACKAGE_DESTINATION}/application.jdl" >& /dev/null

echo "Grid execution script and job JDL file created, you will find these files in the folder  $PACKAGE_DESTINATION"
exit 0
