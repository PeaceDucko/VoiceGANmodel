#! /bin/bash
#SBATCH --partition=long
#SBATCH -N 1 -w mlp03
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH -o ./logs/output.%j.out # STDOUT
#SBATCH --mail-user=sram@science.ru.nl
#SBATCH --time=48:00:00
#
~/VoiceGANmodel/env/bin/python3 ~/VoiceGANmodel/main.py
