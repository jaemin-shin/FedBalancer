# UCI-HAR Dataset

We preprocess the Human Activity Recognition data released by [Davide Anguita et al.](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).
The dataset contains accelerometer and gyroscope data from 30 volunteers performing six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING).

## Setup Instructions

1. Create subdirectory named ```data``` in this directory, and create sub-subdirectory names ```train``` and ```test```. Then, change directory.
```
mkdir -p data
mkdir -p data/train
mkdir -p data/test
cd data
```
2. Download ```UCI HAR Dataset.zip``` file [here](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) to the ```data``` subdirectory.
3. Unzip the data at a ```data``` subdirectory with the following command, and change to the parent directory.
```
unzip 'UCI HAR Dataset.zip'
cd ..
```
4. Run preprocessing.py to generate .json files in ```data/train``` and ```data/test```.
```
python preprocessing.py
```
