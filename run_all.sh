#!/bin/bash

FRACTIONS=(0.1 0.2 0.3 0.5 0.8 1.0)

for frac in "${FRACTIONS[@]}"
do
    echo "Running training with keep_background_fraction=$frac"
    python main.py --keep_background_fraction $frac
done
