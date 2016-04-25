#!/bin/bash
cd "$(dirname "$0")" # working directory
cd ../

mkdir -p build/ logs/ffnn logs/ffdnn
# trash logs/* build/*

# Build the pickles
if [ ! -f ./build/matrix-80.pickle ]; then
	echo "Building Features List"
	python ./src/pickler.py -i ./build/matrixEnron6.txt -o ./build/matrix-80.pickle -t 0.8
fi

for i in `seq 2500 -100 2000`;
do
	for h in `seq $((2*i)) -$((i/5)) $((i))`;
	do
		for model in `echo dnn; echo nn`;
		do
			last=""
			if [ -f logs/ff${model}/${i}-${h}.log ]; then
				last=$(tail -n 1 logs/ff${model}/${i}-${h}.log)
			fi

			if [[ $last != Finished* ]]; then
				if [ $model == "dnn" ]; then
					echo -e "Running Deep Neural Network with ${i} input neurons & ${h} hidden neurons"
				else
					echo -e "Running Neural Network with ${i} input neurons & ${h} hidden neurons"
				fi

				mkdir -p chk/ff${model}/${i}/${h}
				python src/nn/ff_${model}.py --train \
					-f ./build/matrix-80.pickle \
					-i ${i} \
					-n ${h} \
					-b 100 \
					-l 0.03 \
					-z 75 \
					-x 0.005 \
					-s ./chk/ff${model}/${i}/${h}/weights \
				> logs/ff${model}/${i}-${h}.log
			fi
		done
	done
done