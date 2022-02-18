"""End-to-end example for SNN Toolbox.
This script sets up a small CNN using PyTorch, trains it for one epoch on
MNIST, stores model and dataset in a temporary folder on disk, creates a
configuration file for SNN toolbox, and finally calls the main function of SNN
toolbox to convert the trained ANN to an SNN and run it using INI simulator.
"""

import os
import shutil
import inspect
import time

import numpy as np
import torch
from   torch.utils.data import TensorDataset, random_split
import torch.nn as nn
from tensorflow.keras import backend
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser
# from tests.parsing.models.pytorch import Model
from snntoolbox.bin.utils import import_target_sim, update_setup
from importlib import import_module

from data import loadData, generateDataloaders
from models import Model
seed = 528

from shutil import copyfile


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

"""
sigDir = "training_data/siggraph_data"
data, labels = loadData(sigDir)

data = np.expand_dims(data, 1)
print(data.shape)

# using same for train and test, just to see if pipeline works lol
x_train = data[::4]

x_test = data[::2]
y_test = labels[::2]

np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train)
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
"""


# CREATE ANN #
##############

model_name = 'pytorch_FC'
# m = Model()
# torch.save(m.state_dict(), os.path.join(path_wd, model_name + '.pkl'))

src = "C:/Users/taasi/Desktop/trainSNNs/model_dicts/FC_conversion"
state_dict = torch.load(src)
torch.save(state_dict, os.path.join(path_wd, model_name + '.pkl'))


# model_name = 'pytorch_LCN'
# src = "C:/Users/taasi/Desktop/trainSNNs/model_dicts/ex02_LCN_normal_100epoch_k25"
# state_dict = torch.load(src)
# torch.save(state_dict, os.path.join(path_wd, model_name + '.pkl'))


# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
	'path_wd': path_wd,             # Path to model.
									# Path to dataset.
	'dataset_path': 'C:/Users/taasi/Desktop/temp/1642316953.6623368/',         
	'filename_ann': model_name,      # Name of input model.
	'filename_snn': "HI"
}

config['tools'] = {
	'evaluate_ann': False,           # Test ANN on dataset before conversion.
	'normalize': True               # Normalize weights for full dynamic range.
}

config['simulation'] = {
# 	'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
# 	'duration': 50,                 # Number of time steps to run each sample.
# 	'num_to_test': 100,             # How many test samples to run.
# 	'batch_size': 50,               # Batch size for simulation.
# 	'keras_backend': 'tensorflow'   # Which keras backend to use.
}

config['input'] = {
	'model_lib': 'pytorch'          # Input model is defined in pytorch.
}

config['output'] = {
	# 'plot_vars': {                  # Various plots (slows down simulation).
	#     'spiketrains',              # Leave section empty to turn off plots.
	#     'spikerates',
	#     'activations',
	#     'correlation',
	#     'v_mem',
	#     'error_t'}
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
