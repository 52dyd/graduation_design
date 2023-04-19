#!/bin/zsh
python train_main.py >output/$(date +"%Y%m%d-%H:%M:%S").txt
python train_main_single.py >output2/$(date +"%Y%m%d-%H:%M:%S").txt
