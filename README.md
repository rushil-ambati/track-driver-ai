# Track Driver AI

Track Driver AI is a system that allows for easy creation, management and usage of driving recordings and trained AI driver models.

## Installation

### Simulator
Install simulator from [here](https://github.com/udacity/self-driving-car-sim) (look for "Term 1" release for your OS, Version 1 for tracks 1 and 2 or Version 2 for tracks 1 and 3). 

### System
Clone this repository:
```bash
git clone https://github.com/rushil-ambati/track-driver-ai`
```

Then install dependencies:
```bash
cd track-driver-ai
pip install -r requirements.txt
```

## Setup

### Simulator
Run simulator executable, select graphics/control options (joystick is preferable)


### Trainer
Create a blank database:
```bash
cd trainer
python
```
```python
from app import db
db.create_all()
```

Start app:
```bash
python app.py
```

### Driver
```bash
cd driver
```

Start app:
```bash
python driver.py
```

## Usage
### Dataset Recording
Go to the simulator.
1. Enter "Training Mode" on a track you wish to record driving on.
2. Click the record button in the top right.
3. In the file menu, create a new folder inside `trainer/datasets` with your desired dataset name. Select this folder.
4. Start driving around the track.
5. Once you feel as though you have driven enough, stop recording and allow the simulator to capture data into that folder.
6. In advance of the Driving steps, enter "Autonomous Mode" on a track you wish to test the model on.


### Training
Go to the Trainer app (by default, it should be [here](http://127.0.0.1:5000/)).
1. Go to the "Datasets" page and add driving datasets.
2. *(Optional)* Add comments on the quality of driving in that recording or other details for future reference.
3. Chose one to train on and click the "Train" button.
4. *(Optional)* Set training or augmentation parameters - they are set to defaults already.
5. Start training and wait for model fitting to complete.
6. View training output and loss graph (try to recognise under/overfitting)
7. Finish training and go to the "Models" page.

### Driving
Go to the Driver app.
1. Load a model from the `trainer/models` folder.
2. Set the driving speed
3. Begin driving
4. Close the Driver Monitor window to stop driving

### Model Comments
Go to the Trainer app.
1. *(Optional)* Add comments of the quality of the driving of the model on that track or other details for future reference.

## Other
### *(Optional)* Standalone System
These programs run solely on command-line and are best used for testing, experimenting or debugging.

```bash
cd standalone_system
```

Move the dataset folder(s) into the `standalone_system` subfolder.

Set the `data_dir` and `model_name` parameters in the source code to the name of your chosen dataset folder and the desired name of the trained model respectively.

Train:
```bash
python train.py
```

Set the `model_name` parameter to the name of the trained model.

Drive:
```bash
python drive.py
```
