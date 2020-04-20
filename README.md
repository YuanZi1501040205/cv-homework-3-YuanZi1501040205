# Parking Lot Vacancy Detector
__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

A public infrastructure has various parking lots. The parking lots get completely occupied very
often and the public visiting the infrastructure spend too much time looking for a parking space,
unaware that the parking lot is completely occupied. They would like to implement an automated
solution to convey this information by displaying the number of available parking spaces at the
entrance to the parking lot.

### 1. Create train and test datasets for training

Download the github repo:

```sh
$ cd ~
$ git clone https://github.com/shahatuh/cv-homework-3-YuanZi1501040205.git
```
Create the folder as below:
```sh
$ cd cv-homework-3-YuanZi1501040205
~/cv-homework-3-YuanZi1501040205$ mkdir dataset
$ cd dataset
```
Then download the parking slots [dataset](https://www.dropbox.com/sh/5wd3eb35dt7yp2h/AACyHVKO4AQ4oGN6fe-5Wo1Ka?dl=0) to this folder.

Create folders to store the training and testing dataset:
```sh
~/cv-homework-3-YuanZi1501040205$ mkdir train
~/cv-homework-3-YuanZi1501040205$ mkdir test
~/cv-homework-3-YuanZi1501040205$ mkdir train/pos
~/cv-homework-3-YuanZi1501040205$ mkdir train/neg
~/cv-homework-3-YuanZi1501040205$ mkdir test/pos
~/cv-homework-3-YuanZi1501040205$ mkdir test/neg
```


Run Extractor.py code to crop the roi and generate the train and test datasets:
### Arguments explain

| Arguments |Abbreviation |Value |
| ------ |--------|------ |
| path of input |path_input |PATH |
| path of output |path_output |PATH |
| Choose to create train or test dataset |dataset |train, test|
| number of positive samples |numPos |integer |
| number of negative samples |numNeg |integer |
Example Usage: 
python Extractor.py -path_input '/project/kakadiaris/yz/dataset/' -path_output '/project/kakadiaris/yz/dataset/images/' -dataset train -numPos 14000 -numNeg 10000
python Extractor.py -path_input '/project/kakadiaris/yz/dataset/' -path_output '/project/kakadiaris/yz/dataset/images/' -dataset test -numPos 3500 -numNeg 2500

Type these code to create train dataset:
```sh
~/cv-homework-3-YuanZi1501040205$ python Extractor.py -path_input '/cv-homework-3-YuanZi1501040205/dataset/' -path_output '/cv-homework-3-YuanZi1501040205/dataset/' -dataset train -numPos 14000 -numNeg 10000
```
Type these code to create test dataset:
```sh
~/cv-homework-3-YuanZi1501040205$ python Extractor.py -path_input '/project/kakadiaris/yz/dataset/' -path_output '/project/kakadiaris/yz/dataset/images/' -dataset test -numPos 3500 -numNeg 2500
```
### 2. Test the model
