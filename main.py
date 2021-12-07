# import numpy as np
# import matplotlib.pyplot as plt

import torch
from   torchvision import transforms

import torch.nn as nn
import snntorch as snn
import numpy as np
# from   snntorch import spikegen
# from   snntorch import backprop
from   snntorch import surrogate
# import snntorch.spikeplot as splt

# from livelossplot import PlotLosses

from models import LCN, LCNSpiking, LCNSpikingHybrid
from data   import CopyRedChannel, OffSpikes, RateEncodeData, LatencyEncodeData, CopyEncodeLabels
from data   import loadData, generateDataloaders, nSteps
from train  import pipeline, getAtol, testModel

# *******************************************
# Constants

# D2R = np.pi/180
# R2D = 180/np.pi


def normalLCN(device):

	# 100 epochs, batchSize=64
	sigDir = "training_data/siggraph_data"

	# NORMAL ONV **********************************************

	data, labels = loadData(sigDir)
	dataloaders = generateDataloaders(data, labels) 

	model = LCN(14400, 2, 15, 2, 5, True)
	best, last = pipeline(model, epochs=100, lr=1e-3, weight_decay=0, patience=7, atol=1e-5)
	torch.save(last.state_dict(), './model_dicts/linet_normal_100epoch')

	testModel(model, './model_dicts/linet_normal_100epoch', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0.04 2.23  61.0  99.4   100

	# DELTA ONV ***********************************************

	data, labels = loadData(sigDir, 'Delta')
	dataloaders = generateDataloaders(data, labels)

	model = LCN(14400, 2, 15, 2, 5, True)

	best, last = pipeline(model, epochs=100, lr=1e-3, weight_decay=0, patience=7, atol=1e-5)
	torch.save(last.state_dict(), './model_dicts/linet_deltaData_100epoch')

	testModel(model, './model_dicts/linet_deltaData_100epoch', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1  0.5
	#       0    0    0.44  56.7  100   100


def offSpikes(device):

	offSpikes = OffSpikes()

	rate    = RateEncodeData(nSteps, 1, 0)
	latency = LatencyEncodeData(nSteps, 5, 0.01)

	spikeLabels = CopyEncodeLabels(nSteps)
	offRate     = transforms.Compose([offSpikes, rate])

	sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir, 'Delta')
	dataloaders = generateDataloaders(data, labels, xTransform=offRate, yTransform=spikeLabels)

	# SNN, Rate, nSteps=50 , 50 epochs, offspikes
	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_offspikes_50steps_50epochs')

	testModel(m, './model_dicts/linet_spiking_rate_offspikes_50steps_50epochs', dataloaders, device)
	# epoch/atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	# 50          0    0    0     0.56  33.12  99.96


	# SNN, Rate, nSteps=50 , 50 epochs, offspikes, inhibition
	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True, inhibition=True)
	best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_offspikes_inhibition_50steps_50epochs')

	testModel(m, './model_dicts/linet_spiking_rate_offspikes_inhibition_50steps_50epochs', dataloaders, device)
	# epoch/atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	# 50          0    0    0     0.56  33.12  99.96


	# SNN, Rate, nSteps=20 , 100 epochs, offspikes, inhibition, k=20
	m = LCNSpiking(14400, 2, 20, 2, 5, 0.9, 0.8, True, inhibition=True)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_offspikes_inhibition_20steps_100epochs_k20')

	testModel(m, './model_dicts/linet_spiking_rate_offspikes_inhibition_20steps_100epochs_k20', dataloaders, device)
	# epoch/atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	# 50          0    0    0     0.56  33.12  99.96


	# SNN, Rate, nSteps=20 , 100 epochs, offspikes, inhibition, k=25
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True, inhibition=True)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_offspikes_inhibition_20steps_100epochs_k25')

	testModel(m, './model_dicts/linet_spiking_rate_offspikes_inhibition_20steps_100epochs_k25', dataloaders, device)
	# epoch/atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	# 50          0    0    0     0.56  33.12  99.96


	# TODO: try 100 steps
	fastSigmoid = surrogate.fast_sigmoid()
	sigmoid     = surrogate.sigmoid()
	# lso         = surrogate.LeakySpikeOperator()


	# SNN, Rate, nSteps=20 , 100 epochs, offspikes, inhibition, k=25, surrogate
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True, spikeGrad=fastSigmoid)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_offspikes_surrogate_fastSigmoid_20steps_100epochs_k25')

	testModel(m, './model_dicts/linet_spiking_rate_offspikes_surrogate_fastSigmoid_20steps_100epochs_k25', dataloaders, device)
	# epoch/atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	# 50          0    0    0     0.56  33.12  99.96


	# SNN, Rate, nSteps=20 , 100 epochs, offspikes, inhibition, k=25, surrogate
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True, spikeGrad=sigmoid)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_offspikes_surrogate_sigmoid_20steps_100epochs_k25')

	testModel(m, './model_dicts/linet_spiking_rate_offspikes_surrogate_sigmoid_20steps_100epochs_k25', dataloaders, device)
	# epoch/atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	# 50          0    0    0     0.56  33.12  99.96


	# SNN, Rate, nSteps=20 , 100 epochs, offspikes, inhibition, k=25, surrogate
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True, spikeGrad=lso)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_offspikes_surrogate_lso_20steps_100epochs_k25')

	testModel(m, './model_dicts/linet_spiking_rate_offspikes_surrogate_lso_20steps_100epochs_k25', dataloaders, device)
	# epoch/atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	# 50          0    0    0     0.56  33.12  99.96


	# SNN, Rate, nSteps=20 , 100 epochs, offspikes, inhibition, k=25, reset='zero'
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=10, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_offspikes_surrogate_lso_20steps_100epochs_k25')

	testModel(m, './model_dicts/linet_spiking_rate_offspikes_surrogate_lso_20steps_100epochs_k25', dataloaders, device)
	# epoch/atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	# 50          0    0    0     0.56  33.12  99.96

def noOffSpikes(device):
	rate    = RateEncodeData(nSteps, 1, 0)
	latency = LatencyEncodeData(nSteps, 5, 0.01)

	spikeLabels = CopyEncodeLabels(nSteps)

	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	# 50 steps, reset='zero', no offspikes

	# K = 15
	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15')
	testModel(m, './model_dicts/snn_k15', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	      # 0    0    0.08  6     65.88  99.92


	# K = 15, inhibition
	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True, inhibition=True)
	# best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15_inhibition')
	testModel(m, './model_dicts/snn_k15_inhibition', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2   1e-1    0.5
	#       0    0    0.08  3.24   65.36  99.92



	# K = 15, fastSigmoid
	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True, spikeGrad=fastSigmoid)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15_fastSigmoid')
	testModel(m, './model_dicts/snn_k15_fastSigmoid', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0     4.76   65.4   99.96

	# K = 15, inhibition, fastSigmoid
	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True, inhibition=True, spikeGrad=fastSigmoid)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15_inhibition_fastSigmoid')
	testModel(m, './model_dicts/snn_k15_inhibition_fastSigmoid', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0     4.4   64.64  99.96


	# K = 20
	m = LCNSpiking(14400, 2, 20, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k20')
	testModel(m, './model_dicts/snn_k20', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88

	# K = 20, inhibition
	m = LCNSpiking(14400, 2, 20, 2, 5, 0.9, 0.8, True, inhibition=True)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k20_inhibition')
	testModel(m, './model_dicts/snn_k20_inhibition', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88

	# K = 20, fastSigmoid
	m = LCNSpiking(14400, 2, 20, 2, 5, 0.9, 0.8, True, spikeGrad=sigmoid)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k20_sigmoid')
	testModel(m, './model_dicts/snn_k20_sigmoid', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88

	# K = 20, inhibition, fastSigmoid
	m = LCNSpiking(14400, 2, 20, 2, 5, 0.9, 0.8, True, inhibition=True, spikeGrad=sigmoid)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k20_inhibition_sigmoid')
	testModel(m, './model_dicts/snn_k20_inhibition_sigmoid', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88

	# K = 25
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k25')
	testModel(m, './model_dicts/snn_k25', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88

	
	# DIDN'T RUN NEXT 3
	"""
	# K = 25, inhibition
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True, inhibition=True)
	best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k25_inhibition')

	testModel(m, './model_dicts/snn_k15', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	      # 0    0    0.12  5.4   66.96  99.88
	"""

	"""
	# K = 25, fastSigmoid
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True, spikeGrad=fastSigmoid)
	best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k25_fastSigmoid')

	testModel(m, './model_dicts/snn_k15', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88
	"""

	
	"""
	# K = 25, inhibition, fastSigmoid 
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True, inhibition=True, spikeGrad=fastSigmoid)
	best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k25_inhibition_fastSigmoid')

	testModel(m, './model_dicts/snn_k15', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88
	"""


def subtract(device):

	rate    = RateEncodeData(nSteps, 1, 0)
	latency = LatencyEncodeData(nSteps, 5, 0.01)

	spikeLabels = CopyEncodeLabels(nSteps)

	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	# K = 15
	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15_subtract')
	testModel(m, './model_dicts/snn_k15_subtract', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	      # 0    0    0.08  6     65.88  99.92

	# K = 20
	m = LCNSpiking(14400, 2, 20, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k20_subtract')
	testModel(m, './model_dicts/snn_k20_subtract', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88
	
	# K = 25
	m = LCNSpiking(14400, 2, 25, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k25_subtract')
	testModel(m, './model_dicts/snn_k25_subtract', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88


def rateGain(device):

	sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir, 'Delta')

	spikeLabels = CopyEncodeLabels(nSteps)

	"""
	rate    = RateEncodeData(nSteps, 0.25, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15_subtract')
	m.load_state_dict(torch.load('./model_dicts/snn_k15_subtract'))
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	      # 0    0    0.08  6     65.88  99.92
	"""

	"""
	rate    = RateEncodeData(nSteps, 0.5, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15_subtract')
	m.load_state_dict(torch.load('./model_dicts/snn_k15_subtract'))
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	      # 0    0    0.08  6     65.88  99.92
	"""

	"""
	rate    = RateEncodeData(nSteps, 0.75, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15_subtract')
	m.load_state_dict(torch.load('./model_dicts/snn_k15_subtract'))
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	      # 0    0    0.08  6     65.88  99.92
	"""

	"""
	rate    = RateEncodeData(nSteps, 1, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15_subtract')
	m.load_state_dict(torch.load('./model_dicts/snn_k15_subtract'))
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	      # 0    0    0.08  6     65.88  99.92
	"""

	"""
	rate    = RateEncodeData(nSteps, 1.25, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_gain1')	# was saved as gain1
	# m.load_state_dict(torch.load('./model_dicts/snn_k15_subtract_gain1.25'))
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	      # 0    0    0.08  6     65.88  99.92
	"""
	
	rate    = RateEncodeData(nSteps, 1.5, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_gain1.5')
	# m.load_state_dict(torch.load('./model_dicts/snn_k15_subtract_gain1.5'))
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	      # 0    0    0.08  6     65.88  99.92


def firstOrder(device):

	sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir, 'Delta')

	spikeLabels = CopyEncodeLabels(nSteps)
	rate    = RateEncodeData(nSteps, 1.5, 0)

	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 0.99, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_1stOrder_beta0.99')

	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 0.75, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_1stOrder_beta0.75')

	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 0.50, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_1stOrder_beta0.50')

	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 0.25, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_1stOrder_beta0.25')


def hybrid(device):
	sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir, 'Delta')

	# TODO: choose best params from above experiment for rate encoding
	rate    = RateEncodeData(nSteps, 1.5, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate)

	# TODO: choose best params from above experiment for neuron values
	m = LCNSpikingHybrid(4, 14400, 2, 15, 2, 5, 0, 0.99, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_1stOrder_beta0.99_hybrid4')

	m = LCNSpikingHybrid(3, 14400, 2, 15, 2, 5, 0, 0.99, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_1stOrder_beta0.99_hybrid3')

	m = LCNSpikingHybrid(2, 14400, 2, 15, 2, 5, 0, 0.99, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_1stOrder_beta0.99_hybrid2')

	m = LCNSpikingHybrid(1, 14400, 2, 15, 2, 5, 0, 0.99, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_k15_subtract_1stOrder_beta0.99_hybrid1')

def main():

	print("Starting Training Process")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}')

	# *******************************************
	# Dataloaders

	copyRed = CopyRedChannel()

	# SNN data
	offSpikes = OffSpikes()

	rate    = RateEncodeData(nSteps, 1, 0)
	latency = LatencyEncodeData(nSteps, 5, 0.01)

	# SNN labels
	spikeLabels = CopyEncodeLabels(nSteps)

	offRate    = transforms.Compose([offSpikes, rate])
	offLatency = transforms.Compose([offSpikes, latency])

	sigDir = "training_data/siggraph_data"
	
	# data, labels = loadData(sigDir)
	data, labels = loadData(sigDir, 'Delta')

	# data, labels = scaleDownData(data, labels, 0.02)

	# ANN **************************************************************************

	# dataloaders = generateDataloaders(data, labels)                       # 14400 ONV
	# dataloaders = generateDataloaders(data, labels, xTransform=copyRed)   # 43200 ONV

	# SNN **************************************************************************

	# Rate
	# dataloaders = generateDataloaders(data, labels, xTransform=rate)
	# dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	# Latency
	# dataloaders = generateDataloaders(data, labels, xTransform=latency)
	# dataloaders = generateDataloaders(data, labels, xTransform=latency, yTransform=spikeLabels)

	# Off Spikes
	# dataloaders = generateDataloaders(data, labels, xTransform=offRate, yTransform=spikeLabels)
	# dataloaders = generateDataloaders(data, labels, xTransform=offLatency, yTransform=spikeLabels)

	# model = LCN(14400, 2, 15, 2, 5, True)
	# model.load_state_dict(torch.load('linet_deltaData_100epoch'))

	# dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)


	# SNN, Rate, nSteps=20, 100 epochs, batchSize=16
	"""
	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), 'linet_spiking_rate_20steps')

	m.load_state_dict(torch.load('./model_dicts/linet_spiking_rate_20steps'))
	getAtol(m, dataloaders, device, encoding=True)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88
	"""

	"""
	# SNN, Rate, nSteps=50 , 50 epochs

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	best, last = pipeline(m, dataloaders, device, epochs=50, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_50steps_10epochs')

	# m.load_state_dict(torch.load('./model_dicts/linet_spiking_rate_50steps_10epochs'))
	getAtol(last, dataloaders, device, encoding=True)
	# epoch/atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	# 10          0    0    0.12  5.4   66.96  99.88
	# 50          0    0    0     6.3   64.44  99.96
	"""

	rateGain(device)
	firstOrder(device)
	# TODO: change params based on results of above ^^^
	hybrid(device)


if __name__ == "__main__":
	main()