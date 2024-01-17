#!/bin/bash

# Sets up the auto_robots environment for development.
#
# One recommendation is to make a bash function for this script in
# your ~/.bashrc file as follows:
#
# For non-ROS workflows:
#
#  auto_robots() {
#    source ~/workspace/learn_ws/src/auto_robots/setup/source_auto_robots.bash
#  }
#
#  So you can then run this from your Terminal:
#    auto_robots
#

# User variables -- change this to meet your needs
export VIRTUALENV_FOLDER=~/python-virtualenvs/auto_robots
export AUTO_ROBOTS_WS=~/workspace/learn_ws/src/auto_robots

if [ -n "$VIRTUAL_ENV" ]
then
    deactivate
fi

# Activate the Python virtual environment
source $VIRTUALENV_FOLDER/bin/activate

cd $AUTO_ROBOTS_WS
