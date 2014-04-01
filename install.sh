#!/bin/sh

#Select the correct shell configuration file
USER_HOME=$HOME
PROFILE="$USER_HOME/.bashrc" # default
if [ "$(uname)" = "Linux" ]; then

  #Linux case
  PROFILE="$USER_HOME/.bashrc"

else

  #OSX case
  if [ -f "$USER_HOME/.profile" ]; then
    PROFILE="$USER_HOME/.profile"
  else
    PROFILE="$USER_HOME/.bash_profile"
  fi

fi

#Do the installation
if [ "$1" = "local" ]; then
  echo "Installing easea 1.0.3 locally" 

  echo "Exporting and setting environment variables"
  export EZ_PATH="$PWD/"
  export PATH="$PATH:$PWD/bin"
  echo "export EZ_PATH=$EZ_PATH">>$PROFILE
  echo "export PATH=\$PATH:$PWD/bin" >>$PROFILE

else
  echo "Installing easea 1.0.3 in /usr/local/easea" 
  sudo make install > /dev/null

  echo "Exporting and setting environment variables"
  export EZ_PATH="/usr/local/easea/"
  export PATH="$PATH:/usr/local/easea/bin"
  echo "export EZ_PATH=$EZ_PATH">>$PROFILE
  echo "export PATH=\$PATH:/usr/local/easea/bin" >>$PROFILE

fi

