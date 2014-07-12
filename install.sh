#!/bin/sh

#Select the correct shell configuration file
USER_HOME=$HOME
PROFILE="$USER_HOME/.bashrc" # default
if [ "$(uname)" = "Linux" ]; then

  #Linux case
  # Check which sheel the user is using
  # Trim the $SHELL variable to only have the shell name
  SHELL_NAME=`echo $SHELL | awk -F/ '{print $NF}'`

  case "$SHELL_NAME" in
    "sh" )
      PROFILE="$USER_HOME/.profile" ;;
    "bash" )
      PROFILE="$USER_HOME/.bashrc" ;;
    "zsh" )
      PROFILE="$USER_HOME/.zshrc" ;;
    "csh" )
      PROFILE="$USER_HOME/.cshrc" ;;
    "tcsh" )
      PROFILE="$USER_HOME/.tcshrc" ;;
    "ksh" )
      PROFILE="$USER_HOME/.kshrc" ;;
  esac

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
  echo >> $PROFILE
  echo "# EASEA paths for compiler and library">> $PROFILE
  echo "export EZ_PATH=$EZ_PATH">>$PROFILE
  echo "export PATH=\$PATH:$PWD/bin" >>$PROFILE

else
  echo "Installing easea 1.0.3 in /usr/local/easea" 
  sudo make install > /dev/null

  echo "Exporting and setting environment variables"
  export EZ_PATH="/usr/local/easea/"
  export PATH="$PATH:/usr/local/easea/bin"
  echo >> $PROFILE
  echo "# EASEA paths for compiler and library">> $PROFILE
  echo "export EZ_PATH=$EZ_PATH">>$PROFILE
  echo "export PATH=\$PATH:/usr/local/easea/bin" >>$PROFILE

fi

