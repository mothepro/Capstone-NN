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

for i in `seq 2000 100 2500`;
do
    for h in `seq $((i)) $((i/5)) $((2*i))`;
    do
        last_nn=""
        last_dnn=""
        if [ -f logs/ffnn/${i}-${h}.log ]; then
            last_nn=$(tail -n 1 logs/ffnn/${i}-${h}.log)
        fi
        if [ -f logs/ffdnn/${i}-${h}.log ]; then
            last_dnn=$(tail -n 1 logs/ffdnn/${i}-${h}.log)
        fi

        if [[ $last_nn != Finished* ]]; then
            echo -e "Running Neural Network with ${i} input neurons & ${h} hidden neurons"

            mkdir -p chk/ffnn/${i}/${h}
            python src/nn/ff_nn.py --train \
                -f ./build/matrix-80.pickle \
                -i ${i} \
                -n ${h} \
                -b 100 \
                -l 0.03 \
                -z 75 \
                -x 0.005 \
                -s ./chk/ffnn/${i}/${h}/weights \
            > logs/ffnn/${i}-${h}.log
        fi

        if [[ $last_dnn != Finished* ]]; then
            echo -e "Running Deep Neural Network with ${i} input neurons & ${h} hidden neurons"

            mkdir -p chk/ffdnn/${i}/${h}
            python src/nn/ff_dnn.py --train \
                -f ./build/matrix-80.pickle \
                -i ${i} \
                -n ${h} \
                -b 100 \
                -l 0.05 \
                -z 75 \
                -x 0.005 \
                -s ./chk/ffdnn/${i}/${h}/weights \
            > logs/ffdnn/${i}-${h}.log
        fi
    done
done