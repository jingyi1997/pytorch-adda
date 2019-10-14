"""Params for ADDA."""

# params for dataset and data loader
data_root = "data/raw"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 128
image_size = 64

# params for source dataset
src_dataset = "MNIST"
src_dataset_root = "data/processed/training"
src_dataset_list = "data/processed/training_list.txt"
src_dataset_eval_root = "data/processed/testing"
src_dataset_eval_list = "data/processed/testing_list.txt"
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = "USPS"
tgt_dataset_root = "data/raw/training"
tgt_dataset_list = "data/raw/training_list.txt"
tgt_dataset_eval_root = "data/raw/testing"
tgt_dataset_eval_list = "data/raw/testing_list.txt"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-1600.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
num_epochs_pre = 40
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 2000
log_step = 10
save_step = 400
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
