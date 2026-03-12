#!/bin/bash

TEXTS=("road" "way" "walkway")

for TEXT in "${TEXTS[@]}"
do
    python3 run_seem_image.py --text "$TEXT"
done