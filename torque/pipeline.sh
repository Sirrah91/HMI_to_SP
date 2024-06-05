#!/bin/bash

export PATH=$PATH:/opt/torque/bin/

base_dir="/nfshome/david/NN/"
WD="${base_dir}/python_compiled/"
LD="${base_dir}/log/"

mkdir -p ${LD}

# "--num_repeats int": number of trained models (default 1)
# "--train": whether to train or evaluate a new model (default True)
# "--tune_hp": whether to tune hyperparameters (default False; ignores other options)
# "--add_mem float": additional memory due to data (default 0.)
# "--model_to_retrain str": filename of saved weights in "HMI_to_SOT" directory that should be retrained (default "")

### defaults of the followings are in NN_HP.py ###
# "--model_type str": CNN or CNN_sep
# "--num_residuals int": number of residual blocks
# "--num_nodes int": number of convolution kernels
# "--kern_size int": size of the convolution kernels
# "--kern_pad str": valid or same
# "--input_activation str": activation function except for the very last one

# "--dropout_input_hidden float": dropout between the input and the first hidden layer
# "--dropout_residual_residual float": dropout between the residual blocks
# "--dropout_hidden_output float": dropout between the last hidden layer and the output
# "--L1_trade_off float": trade-off for the L1 regularisation
# "--L2_trade_off float": trade-off for the L2 regularisation
# "--optimizer str": optimization scheme
# "--learning_rate float": learning rate in the optimization scheme
# "--batch_size int": batch size
# "--bs_norm_before_activation": apply batch size normalization before or after activation function

# "--loss_type str": which loss function to use
# "--metrics str str ...": which metrics function to use
# "--alpha float": trade-off between continuum (I) and magnetic field (B) misfits (alpha x I + B)
# "--c float": parameter in Cauchy loss function
# "--num_epochs int": maximum number of epochs for training (not hp tuning)

SETTINGS="--train --model_to_retrain weights_1111_20240513105441.h5 --add_mem 8.0 --learning_rate 0.00001 --num_epochs 300"

cd ${WD} || exit
SIZE=($(./job_sizes ${SETTINGS}))
echo "NN job size: ${SIZE[0]} GB"
TD="${base_dir}/torque/"
cd ${TD} || exit

n_jobs=1
for _ in $(seq $n_jobs); do
    qsub -l mem="${SIZE[0]}"gb -v WD=${WD},LD=${LD},SETTINGS="${SETTINGS}" HMI_to_SOT.pbs
    sleep 2
done
