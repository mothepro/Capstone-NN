#!/bin/bash
cd "$(dirname "$0")" # working directory
cd ../

mkdir -p build/ logs/ffnn chk/ffnn logs/ffdnn chk/ffdnn
# trash logs/* build/*

# Build the pickles
if [ ! -f ./build/matrix.pickle ]; then
    echo "Building Features List"
    python ./src/pickler.py -i ./build/matrixEnron6.txt -o ./build/matrix.pickle -t 0.8
fi

for i in `seq 100 100 2500`;
do
    for h in `seq $((i)) $((i/10)) $((2*i))`;
    do
        last_nn=$(tail -n ./logs/ffnn/${i}-${h}.log)
        last_dnn=$(tail -n ./logs/ffdnn/${i}-${h}.log)

        if [[ $last_nn == Finished* ]]; then
            echo -e "Running Neural Network with ${i} input neurons & ${h} hidden neurons"
            python src/nn/ff_nn.py \
                -f ./build/matrix.pickle \
                -i ${i} \
                -n ${h} \
                -b 100 \
                -l 0.005 \
                -z 75 \
                -x 0.01 \
                -s ./chk/ffnn/${i}-${h} \
            > ./logs/ffnn/${i}-${h}.log
        fi

        if [[ $last_dnn == Finished* ]]; then
            echo -e "Running Deep Neural Network with ${i} input neurons & ${h} hidden neurons"
            python src/nn/ff_dnn.py \
                -f ./build/matrix.pickle \
                -i ${i} \
                -n ${h} \
                -b 100 \
                -l 0.005 \
                -z 75 \
                -x 0.01 \
                -s ./chk/ffdnn/${i}-${h} \
            > ./logs/ffdnn/${i}-${h}.log
        fi
    done
done