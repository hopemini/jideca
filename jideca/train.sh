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

## Save model per 100 epochs
#echo 'jideca semantic_annotations training..'
#python train.py -t semantic_annotations -gid 0 -e 2500 -js -beta 10 -gamma 0.2 --off_scheduling -p 100

echo 'jideca semantic_annotations training..'
python train.py -t semantic_annotations -gid 0 -e 2500 -js -beta 10 -gamma 0.2 --off_scheduling

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
