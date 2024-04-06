# Red-light Running Detection

Build a system to detect red-light running violation using YOLOv8 and Deep Learning.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#license)

## Installation
Firstly, clone this repository and save it in your PC.
#### Clone repository
```http
  git clone 
```
#### Python 3.7.16
Secondly, install Python version 3.7.16
```http
  pip install python==3.7.16
```
#### CUDA 11.6
In order to run object detection model using NVIDIA GPU, CUDA need to be installed. The version of CUDA used for this project is 11.6. (For NVIDIA PC only)
```http
https://developer.nvidia.com/cuda-11-6-0-download-archive
```

#### Install packages in requirements.txt
All of the packages used in this project is in the requirements.txt file. You the following comand to install these packages:
 ```http
pip install -r requirements.txt 
   ```

## Usage
After installing all dependencies, you will be able to run the program by running the runs.py.
#### Run the program
```http
python runs.py
   ```

#### User Interface Buttons
- **Selection MP4 File**: Open MP4 file.
- **Pause/Clear Video**: Pause or clear the detection process.
- **Export to PDF/CSV**: Export recorded violation under PDF or Excel spreadsheet format.


## Acknowledgements
 - [Ultralytics](https://github.com/ultralytics/ultralytics)
 - [DeepSort](https://github.com/nwojke/deep_sort)

