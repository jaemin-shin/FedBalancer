# UCI-HAR Dataset

We preprocess the Human Activity Recognition data released by [Davide Anguita et al.](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).
The dataset contains accelerometer and gyroscope data from 30 volunteers performing six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING).

## Setup Instructions

1. Run ```mkdir data``` to create directory named data in this directory.
2. Download ```UCI HAR Dataset.zip``` file [here](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) and unzip the data into a ```data``` subfolder in this directory.
3. With the data in the appropriate directory, run build the training vocabulary by running ```python build_vocab.py --data-dir ./data/train --target-dir vocab```. 
