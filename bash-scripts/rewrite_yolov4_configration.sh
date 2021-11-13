#!/bin/zsh
CLAESES=3
FIL=$(( (CLAESES+5)*3 ))
mv yolov4-tiny.cfg yolov4-tiny.cfg.bak
cat yolov4-tiny.cfg.bak | sed -e "s/filters=255/filters=${FIL}/g" -e "s/classes=80/classes=${CLAESES}/g" > yolov4-tiny.cfg
rm *bak
