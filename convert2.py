"""End-to-end example for SNN Toolbox.

This script sets up a small CNN using PyTorch, trains it for one epoch on
MNIST, stores model and dataset in a temporary folder on disk, creates a
configuration file for SNN toolbox, and finally calls the main function of SNN
toolbox to convert the trained ANN to an SNN and run it using INI simulator.
"""

# """
import os
import shutil
import inspect
import time

import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras import backend
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser
from tests.parsing.models.pytorch import Model

from importlib import import_module
from snntoolbox.bin.utils import update_setup

# Pytorch to Keras parser needs image_data_format == channel_first.
backend.set_image_data_format('channels_first')

# WORKING DIRECTORY #
#####################

# Define path where model and output files will be stored.
# The user is responsible for cleaning up this temporary directory.
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(time.time())))
os.makedirs(path_wd)

# GET DATASET #
###############

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input so we can train ANN with it.
# Will be converted back to integers for SNN layer.
x_train = x_train / 255
x_test = x_test / 255

# Add a channel dimension.
axis = 1 if backend.image_data_format() == 'channels_first' else -1
x_train = np.expand_dims(x_train, axis)
x_test = np.expand_dims(x_test, axis)

# One-hot encode target vectors before saving for use in toolbox.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Save dataset so SNN toolbox can find it.
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
# SNN toolbox will not do any training, but we save a subset of the training
# set so the toolbox can use it when normalizing the network parameters.
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])

# Pytorch doesn't support one-hot labels, so we undo it for training the ANN.
y_train = np.argmax(y_train, 1)
y_test = np.argmax(y_test, 1)


class PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x, self.target[index]

    def __len__(self):
        return len(self.data)


trainset = torch.utils.data.DataLoader(PytorchDataset(x_train, y_train),
                                       batch_size=64)
testset = torch.utils.data.DataLoader(PytorchDataset(x_test, y_test),
                                      batch_size=64)

# CREATE ANN #
##############

# This section creates a CNN using pytorch, and trains it with backpropagation.
# There are no spikes involved at this point.

# Create pytorch model from definition in separate script.
model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train model with backprop.
acc = 0
for epoch in range(3):
    for i, (xx, y) in enumerate(trainset):
        optimizer.zero_grad()
        outputs = model(xx)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    total = 0
    correct = 0
    with torch.no_grad():
        for xx, y in testset:
            outputs = model(xx)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc = correct / total

print("Test accuracy: {:.2%}".format(acc))

# Store weights so SNN Toolbox can find them.
model_name = 'pytorch_cnn'
torch.save(model.state_dict(), os.path.join(path_wd, model_name + '.pkl'))

# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'dataset_path': path_wd,        # Path to dataset.
    'filename_ann': model_name,      # Name of input model.
    'filename_snn': "HI"
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'normalize': True               # Normalize weights for full dynamic range.
}

config['simulation'] = {
    'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
    'duration': 50,                 # Number of time steps to run each sample.
    'num_to_test': 100,             # How many test samples to run.
    'batch_size': 50,               # Batch size for simulation.
    'keras_backend': 'tensorflow'   # Which keras backend to use.
}

config['input'] = {
    'model_lib': 'pytorch'          # Input model is defined in pytorch.
}

config['output'] = {
    'plot_vars': {                  # Various plots (slows down simulation).
        'spiketrains',              # Leave section empty to turn off plots.
        'spikerates',
        'activations',
        'correlation',
        'v_mem',
        'error_t'}
}

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# Need to copy model definition over to ``path_wd`` (needs to be in same dir as
# the weights saved above).
source_path = inspect.getfile(Model)
shutil.copyfile(source_path, os.path.join(path_wd, model_name + '.py'))

# RUN SNN TOOLBOX #
###################

# main(config_filepath)

# """

"""
import h5py
from tests.parsing.models.pytorch import Model

m = Model()

print(m.state_dict)

filename = "C:/Users/taasi/Desktop/temp/1642225641.041376/HI.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])

    print(data)

"""

# """
config = update_setup(config_filepath)

queue = None

# Instantiate an empty spiking network
# target_sim = import_target_sim(config)
# spiking_model = target_sim.SNN(config, queue)


 # __________________________ LOAD MODEL _____________________________ #

model_lib = import_module('snntoolbox.parsing.model_libs.' +
                          config.get('input', 'model_lib') +
                          '_input_lib')
input_model = model_lib.load(config.get('paths', 'path_wd'),
                             config.get('paths', 'filename_ann'))


# ____________________________ PARSE ________________________________ #

print("Parsing input model...")
model_parser = model_lib.ModelParser(input_model['model'], config)
model_parser.parse()
parsed_model = model_parser.build_parsed_model()


# ___________________________ NORMALIZE _____________________________ #

if config.getboolean('tools', 'normalize') and not is_stop(queue):
    normalize_parameters(parsed_model, config, **normset)

# Evaluate parsed model.
if config.getboolean('tools', 'evaluate_ann') and not is_stop(queue):
    print("Evaluating parsed model on {} samples...".format(
        num_to_test))
    score = model_parser.evaluate(
        config.getint('simulation', 'batch_size'),
        num_to_test, **testset)
    results = [score[1]]

# Write parsed model to disk
parsed_model.save(str(
    os.path.join(config.get('paths', 'path_wd'),
                 config.get('paths', 'filename_parsed_model') +
                 '.h5')))


# _____________________________ CONVERT _________________________________ #

if config.getboolean('tools', 'convert') and not is_stop(queue):
    if parsed_model is None:
        try:
            parsed_model = load(
                config.get('paths', 'path_wd'),
                config.get('paths', 'filename_parsed_model'),
                filepath_custom_objects=config.get(
                    'paths', 'filepath_custom_objects'))['model']
        except FileNotFoundError:
            print("Could not find parsed model {} in path {}. Consider "
                  "setting `parse = True` in your config file.".format(
                    config.get('paths', 'path_wd'),
                    config.get('paths', 'filename_parsed_model')))

    spiking_model.build(parsed_model, **testset)

    # Export network in a format specific to the simulator with which it
    # will be tested later.
    spiking_model.save(config.get('paths', 'path_wd'),
                       config.get('paths', 'filename_snn'))

# """
