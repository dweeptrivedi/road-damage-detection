# road damage detection

## Table of contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Running the code](#running-the-code)
- [Authors](#authors)

## Prerequisites

You need to install:
- [Python3](https://www.python.org/downloads/)

## Quick-start
1. To start using the vehicle-detector you need to clone the repo: `https://github.com/dweeptrivedi/road-damage-detection.git`

2. Execute `./build_darknet.sh`
  - This command will build Yolo library for CPU and GPU

## Running the code

Testing:

- run `python3 detector.py` for default network

- run `python3 detector.py --help` for commandline options

- Best possible combination examples:
    - `python3 detector.py --input-file="test.txt" --approach="one-phase" --yolo=45000 --nms=0.999 --thresh=0.1 --gpu=False`
    - `python3 detector.py --input-file="test.txt" --approach="augmented" --yolo=60000 --nms=0.999 --thresh=0.1 --gpu=False`
    - `python3 detector.py --input-file="test.txt" --approach="cropped" --yolo=60000 --nms=0.999 --thresh=0.1 --gpu=False`


## Authors:
* **Dweep Trivedi** - Please give me your feedback: dweeptrivedi1994@gmail.com
