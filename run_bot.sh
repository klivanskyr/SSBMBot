#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ssbmbot

# Determine which script to run (default to play_bot.py)
SCRIPT="${1:-play_bot.py}"

# Remove the script name from arguments if it was provided
if [ "$1" = "play_bot.py" ] || [ "$1" = "example.py" ]; then
    shift
fi

# Run the script with proper paths
python "$SCRIPT" \
    --dolphin_executable_path ~/.config/"Slippi Launcher"/netplay/Slippi_Online-x86_64.AppImage \
    --iso /home/klivanskyr/courses/csci2470/SSBMBot/Melee.iso \
    "$@"
