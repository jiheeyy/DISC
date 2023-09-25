#!/bin/bash
unset XDG_RUNTIME_DIR
NODEIP=$(hostname -i)
NODEPORT=$(( $RANDOM + 1024))
echo "ssh command: ssh -N -L 8888:$NODEIP:$NODEPORT `jiheeyou`@fe01.ai.cs.uchicago.edu"
jupyter-notebook --ip=$NODEIP --port=$NODEPORT --no-browser
