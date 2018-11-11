# road damage detection

## Table of contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Running the code](#running-the-code)

## Prerequisites

You need to install:
- [Python3](https://www.python.org/downloads/)

## Quick-start
1. Clone the repo: `https://github.com/dweeptrivedi/road-damage-detection.git`

2. Execute `./build_darknet.sh`

## Running the code

### Testing:

- `test.txt` should contain list of image paths that needs detection.

- run `python3 detector.py` for default network.

- Best combination examples:
    - `python3 detector.py --input-file="test.txt" --approach="one-phase" --yolo=45000 --nms=0.999 --thresh=0.1`
    - `python3 detector.py --input-file="test.txt" --approach="augmented" --yolo=60000 --nms=0.999 --thresh=0.1`
    - `python3 detector.py --input-file="test.txt" --approach="cropped" --yolo=60000 --nms=0.999 --thresh=0.1`

### Training:

- Yolo: For training Yolo network for road damage detection task, please follow steps mentioned in [Yolo](https://pjreddie.com/darknet/yolo/) website.
