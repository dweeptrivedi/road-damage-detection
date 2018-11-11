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
    - `python3 detector.py --input-file="test.txt" --approach="one-phase" --yolo=45000 --nms=0.999 --thresh=0.01`
    - `python3 detector.py --input-file="test.txt" --approach="augmented" --yolo=60000 --nms=0.999 --thresh=0.0508`
    - `python3 detector.py --input-file="test.txt" --approach="cropped" --yolo=60000 --nms=0.999 --thresh=0.15`
    
- command-line option details:

| option |  possible values | default values | Notes |
| --- | --- | --- | --- |
| --input-file | location of the file | "test.txt" | contains list of image paths that needs detection. |
| --approach | ["one-phase", "cropped", "augmented"] | "one-phase" |name of the approach (as described in paper) |
| --yolo | int | 45000 | iteration number of Yolo weight file. I have uploaded one weight file (best performance on test dataset) for each approach  |
| --nms | float value in [0,1] | 0.45 | nms threshold value for dropping overlapping predictions |
| --thresh | float value in [0,1] | 0.1 | confidence threshold value for predictions |
| --gpu | boolean | False | whether to use GPU or not. Set this option only if `build_darknet.sh` returned successful build for GPU |

### Training:

- Yolo: For training Yolo network for road damage detection task, please follow steps mentioned in [Yolo](https://pjreddie.com/darknet/yolo/) website.
