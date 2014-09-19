# From http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SCRIPT_PATH="${BASH_SOURCE[0]}";
if([ -h "${SCRIPT_PATH}" ]) then
  while([ -h "${SCRIPT_PATH}" ]) do SCRIPT_PATH=`readlink "${SCRIPT_PATH}"`; done
fi
pushd . > /dev/null
cd `dirname ${SCRIPT_PATH}` > /dev/null
SCRIPT_PATH=`pwd`;
popd  > /dev/null


OCELOT_ROOT=$SCRIPT_PATH

export OCELOT_INSTALL_PATH=$OCELOT_ROOT/build_local

export PATH="$OCELOT_INSTALL_PATH/bin:$PATH"
[ -n "$LD_LIBRARY_PATH" ] && \
  export LD_LIBRARY_PATH="$OCELOT_INSTALL_PATH/lib:$LD_LIBRARY_PATH" || \
  export LD_LIBRARY_PATH="$OCELOT_INSTALL_PATH/lib"
