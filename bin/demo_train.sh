#!/bin/bash
cd "$(dirname "$0")/../.." # working directory
mkdir -p emails/build emails/chk emails/demo
i=2500
h=3750

if [ ! -f ./emails/build/train.features ]; then # Make feature matrix for emails in emails/
	echo "Building Features List"

	# BaseSet
	#mono FM/FEATURES/FEATURES/bin/Release/FEATURES.exe -f ./emails/demo/train/ -df -l ./emails/demo/train/train.labels > ./emails/build/train.features

	# Enron
	mono FM/FEATURES/FEATURES/bin/Release/FEATURES.exe -f ./emails/demo/ham/  -df -lh 1 > ./emails/build/ham.features
	mono FM/FEATURES/FEATURES/bin/Release/FEATURES.exe -f ./emails/demo/spam/ -df -lh 0 > ./emails/build/spam.features
	cat emails/build/ham.features emails/build/spam.features > emails/build/train.features
fi

if [ ! -f ./emails/build/train.pickle ]; then # Build the pickles
	echo "Building Features Matrix"
	python3 ./Capstone-NN/src/pickler.py -i ./emails/build/train.features -o ./emails/build/train.pickle --train 0.8
fi

echo -e "Training Neural Network with ${i} input neurons & ${h} hidden neurons"
python3 ./Capstone-NN/src/nn/ff_nn.py --train \
	-f ./emails/build/train.pickle \
	-i ${i} \
	-n ${h} \
	-b 100 \
	-l 0.03 \
	-z 75 \
	-x 0.005 \
	-s ./emails/chk/${i}-${h}-weights \
> ./emails/train.log
