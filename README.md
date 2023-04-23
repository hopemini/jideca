# JIDECA
Jointly Improved Deep Embedded Clustering for Android activity
<!--
This repository contains source code for paper [JIDECA: Jointly Improved Deep Embedded Clustering for Android activity](https://ieeexplore.ieee.org/abstract/document/10066814)
```
@inproceedings{choi2023jideca,
  title={JIDECA: Jointly Improved Deep Embedded Clustering for Android activity},
  author={Choi, Sungmin and Seo, Hyeon-Tae and Han, Yo-Sub},
  booktitle={2023 IEEE International Conference on Big Data and Smart Computing (BigComp)},
  pages={105--112},
  year={2023},
  organization={IEEE},
  doi={10.1109/BigComp57234.2023.00025}
}
```
-->

## Data processing files
Download [Link](https://drive.google.com/file/d/1wacGwcTHUPWZ-c9mouq6AVUepagiVKQo/view?usp=sharing) and extract it.
```
$ tar xzvf data_processing.tar.gz
```

## Create train and test dataset
(We already upload dataset. Please, check [code](https://github.com/hopemini/jideca/tree/main/jideca/data).)

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
If necessary, change **num_workers** (line 137) and **batch_size** (line 67) in train.py.
The default values are 64 and 2400, respectively.
```
option:
-t: data types [real, semmantic_annotations]
-c: number of classes (default: 34)
-e: number of epochs (default: 5000)
-gid: GPU id (default: 0)
-r: resume
-js: use JS Divergence (use alignment loss)
-beta: hyperparameter for reconstruction term
-gamma: hyperpapameter for alignment term
-l : hyperparameter for lambda
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

## DEC
#echo 'dec re training..'
#python train.py -t real -gid 0 -e 2500 -l 1 -beta 0  --off_scheduling -c 23

#echo 'dec se training..'
#python train.py -t semantic_annotations -gid 0 -e 2500 -l 1 -beta 0  --off_scheduling -c 23

#echo 'dec dnn training..'
#python train.py -t real -gid 0 -e 2500 -l 0 -beta 0  --off_scheduling -c 23

## IDEC
#echo 'idec re training..'
#python train.py -t real -gid 0 -e 2500 -l 1 -beta 1  --off_scheduling -c 23

#echo 'idec se training..'
#python train.py -t semantic_annotations -gid 0 -e 2500 -l 1 -beta 1  --off_scheduling -c 23

#echo 'idec dnn training..'
#python train.py -t real -gid 0 -e 2500 -l 0 -beta 1  --off_scheduling -c 23
```

Modify train.sh to suit your experiment and run it.
```
. ./train.sh
```

## Evaluation
```
cd ../evaluation
```
Please, change proper number and names in the test_evaluation.py.
```
103     for _iter in range(10):
104         run("jideca_b10g01re_34", "34", _iter)
105         run("jideca_b10g01se_34", "34", _iter)
106         run("jideca_b10g02re_34", "34", _iter)
107         run("jideca_b10g02se_34", "34", _iter)
108         run("jideca_b10g05re_34", "34", _iter)
109         run("jideca_b10g05se_34", "34", _iter)
110         run("jideca_b10g1re_34", "34", _iter)
111         run("jideca_b10g1se_34", "34", _iter)
112         run("jideca_b10g10re_34", "34", _iter)
113         run("jideca_b10g10se_34", "34", _iter)
```
Then,
```
$ . ./test_evaluation.sh
```

## Evaluation (including purity)
Same process as [Evaluation](https://github.com/hopemini/jideca#evaluation), but use **test_evaluation_purity.py** and **test_evaluation_purity.sh**.
