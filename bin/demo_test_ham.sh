#!/bin/bash
cd "$(dirname "$0")/../.." # working directory
mkdir -p emails/build emails/chk emails/demo
i=2500
h=3750

if [ ! -f ./emails/build/test_ham.features ]; then # Make feature matrix for emails in emails/
	echo "Building Features List"
	mono FM/FEATURES/FEATURES/bin/Release/FEATURES.exe -f ./emails/demo/test/ham/ -df > ./emails/build/test_ham.features
fi

if [ ! -f ./emails/build/test_ham.pickle ]; then # Build the pickles
	echo "Building Features Matrix"
	python3 ./Capstone-NN/src/pickler.py -i ./emails/build/test_ham.features -o ./emails/build/test_ham.pickle
fi

echo -e "Testing Neural Network with ${i} input neurons & ${h} hidden neurons"
python3 ./Capstone-NN/src/nn/ff_nn.py --test \
	-f ./emails/build/test_ham.pickle \
	-i ${i} \
	-n ${h} \
	-s ./emails/chk/
