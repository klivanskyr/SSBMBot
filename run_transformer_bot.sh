#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ssbmbot

# Run the transformer bot with proper paths
python play_bot.py \
    --dolphin_executable_path ~/.config/"Slippi Launcher"/netplay/Slippi_Online-x86_64.AppImage \
    --iso /home/klivanskyr/courses/csci2470/SSBMBot/Melee.iso \
    --checkpoint checkpoints/best_model.pt \
    --port 1 \
    --opponent 2 \
    --character FOX \
    --stage YOSHIS_STORY \
    "$@"

# ./run_transformer_bot.sh --character MARTH --opponent_character FALCO --cpu_level 9 --stage BATTLEFIELD
