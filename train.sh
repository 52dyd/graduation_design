#!/bin/zsh
python train_main.py >output/$(date +"%Y-%m-%d %H:%M:%S").txt 2>output/$(date +"%Y-%m-%d %H:%M:%S").err
