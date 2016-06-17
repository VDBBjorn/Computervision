#!/bin/bash
rm -rf CMake*
openCvCmake.bash maxSpeed
./maxSpeed $1
subl ./Dataset/evaluate.py
nemo ./outputframes/$1