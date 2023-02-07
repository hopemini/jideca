# JIDECA
Jointly Improved Deep Embedded Clustering for Android activity

## Data processing files
Download [Link](https://drive.google.com/file/d/1wacGwcTHUPWZ-c9mouq6AVUepagiVKQo/view?usp=sharing) and extract it.
```
$ tar xzvf data_processing.tar.gz
```

## Create train and test dataset
If necessary, change **num_workers** (line 28 and 40) and **batch_size** (line 50) in data_processing.py.
The default values are 8 and 2400, respectively.
```
$ cd jideca
$ python data_processing.py
```

output
```
data/train_re.pkl
data/train_se.pkl
data/test_re_23.pkl
data/test_re_34.pkl
data/test_se_23.pkl
data/test_se_34.pkl
```

## Train
If necessary, change **num_workers** (line 130) and **batch_size** (line 60) in train.py.
The default values are 8 and 2400, respectively.
```
option:
-t: data types [real, semmantic_annotations]
-c: number of classes (default: 34)
-e: number of epochs
-r: resume
-js: use JS Divergence
-beta: hyperparameter for reconstruction term
-gamma: hyperpapameter for alignment term
--off_scheduling
```

```
$ cat train.sh
## JIDECA
#echo 'jideca real training..'
#python train.py -t real -gid 0 -e 2500 -js --off_scheduling

## JIDECA, change beta (reconstruction), gamma (alignment)
#echo 'jideca real training..'
#python train.py -t real -gid 0 -e 2500 -js -beta 1 -gamma 1 --off_scheduling

## JIDECA w/o alingment (== multimodal IDEC), no -js option
#echo 'multimodal idec real training..'
#python train.py -t real -gid 0 -e 2500 -gamma 0 --off_scheduling

## JIDECA w/o reconstruction (== multimodal DEC w/ alignment)
#echo 'multimodal dec w/ alignment real training..'
#python train.py -t real -gid 0 -e 2500 -js -beta 0 --off_scheduling

## JIDECA w/o reconstruction and alignment (== multimodal DEC)
#echo 'multimodal dec real training..'
#python train.py -t real -gid 0 -e 2500 -beta 0 -gamma 0 --off_scheduling

echo 'jideca semantic_annotations training..'
python train.py -t semantic_annotations -gid 0 -e 2500 -js --off_scheduling
```

Modify train.sh to suit your experiment and run it.
```
. ./train.sh
```
