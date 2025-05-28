# Tempography Video Temporal Colorization Tool

## Overview
Tempography is a tool for temporal colorization of videos, designed to process either a single video or all videos in a folder. It supports both manual and automatic ROI (Region of Interest) selection, background removal using various methods, and stacking of frames with rainbow colormap application. The results are saved as PNG images.

## Features
- Batch process all videos in a folder or a single video
- Manual or automatic ROI (Region of Interest) selection
- Automatic background calculation (mode/median/average)
- Background removal and stacking of all frames with rainbow colormap
- Results automatically saved as PNG images

## Installation Requirements
- Python 3.7+
- opencv-python
- numpy
- scipy

### Install dependencies
```bash
pip install opencv-python numpy scipy
```

## Usage
### Process all videos in a folder with manual ROI selection
```bash
python main.py --folder <video_folder> --show
```

### Process all videos in a folder with the same ROI
```bash
python main.py --folder <video_folder> --roi 100,100,300,300
```

### Process a single video
```bash
python main.py --file <video_path> --roi 100,100,300,300
```

### Command Line Arguments
- `--folder`      Path to the folder containing videos
- `--file`        Path to a single video file
- `--roi`         Specify ROI in the format: x,y,w,h
- `--bg_method`   Background calculation method (`mode`/`medi`/`avg`)
- `--max_bg_frames` Maximum number of frames for background calculation
- `--stack_method` Stacking method (`mean`/`median`/`max`)
- `--frame_skip`  Number of frames to skip at the beginning (default: 10000)
- `--show`        Show the result image
- `--save_dir`    Output directory for results

## License
This project is licensed under the MIT License. 