#!/bin/bash

USER_TO_CHECK="lorenzoquerol"

# Check if the username is provided
if [ -z "$USER_TO_CHECK" ]; then
    echo "Please provide a username as an argument."
    exit 1
fi

# Get PPIDs of zombie processes of the specified user
for ppid in $(ps -u $USER_TO_CHECK -o ppid,stat | awk '$2 ~ /^Z/ {print $1}' | sort -u); do
    # First, try to send SIGCHLD to the parent
    kill -s SIGCHLD $ppid
    sleep 1

    # Check if zombies of that parent still exist
    if ps -u $USER_TO_CHECK -o ppid,stat | awk -v parent="$ppid" '$1 == parent && $2 ~ /^Z/ {print $1}' | grep -q "$ppid"; then
        # If zombies still exist, kill the parent
        echo "Killing parent process $ppid as zombies remain."
        kill -9 $ppid
    fi
done