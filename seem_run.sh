#!/bin/bash

TEXTS=("traversable area" "walkable area" "runnable area" "road" "way" "walkway" "sidewalk" "street" "pavement")

for TEXT in "${TEXTS[@]}"
do
    python3 run_seem_image.py --text "$TEXT"
done