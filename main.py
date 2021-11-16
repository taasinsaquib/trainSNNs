# import numpy as np
# import matplotlib.pyplot as plt

import torch
from   torchvision import transforms

# import snntorch as snn
# from   snntorch import spikegen
# from   snntorch import backprop
# from   snntorch import surrogate
# import snntorch.spikeplot as splt

# from livelossplot import PlotLosses

from models import LCN, LCNSpiking
from data   import CopyRedChannel, OffSpikes, RateEncodeData, LatencyEncodeData, CopyEncodeLabels
from data   import loadData, generateDataloaders, nSteps
from train  import pipeline, getAtol

# *******************************************
# Constants

# D2R = np.pi/180
# R2D = 180/np.pi


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
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	# Latency
	# dataloaders = generateDataloaders(data, labels, xTransform=latency)
	# dataloaders = generateDataloaders(data, labels, xTransform=latency, yTransform=spikeLabels)

	# Off Spikes
	# dataloaders = generateDataloaders(data, labels, xTransform=offRate, yTransform=spikeLabels)
	# dataloaders = generateDataloaders(data, labels, xTransform=offLatency, yTransform=spikeLabels)

	# 100 epochs, normal vs delta data on normal LCN, batchSize=64
	"""

	print("Data Loaded")

	model = LCN(14400, 2, 15, 2, 5, True)

	best, last = pipeline(model, epochs=100, lr=1e-3, weight_decay=0, patience=7, atol=1e-5)
	torch.save(last.state_dict(), 'linet_normal_100epoch')

	getAtol(model)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0.04 2.23  61.0  99.4   100

	data, labels = loadData(sigDir, 'Delta')
	dataloaders = generateDataloaders(data, labels)                       # 14400 ONV

	model = LCN(14400, 2, 15, 2, 5, True)

	best, last = pipeline(model, epochs=100, lr=1e-3, weight_decay=0, patience=7, atol=1e-5)
	torch.save(last.state_dict(), 'linet_deltaData_100epoch')

	getAtol(model, dataloaders)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1  0.5
	#       0    0    0.44  56.7  100   100
	"""

	# model = LCN(14400, 2, 15, 2, 5, True)
	# model.load_state_dict(torch.load('linet_deltaData_100epoch'))

	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	"""
	# SNN, Rate, nSteps=20, 100 epochs, batchSize=16

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	# best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	# torch.save(last.state_dict(), 'linet_spiking_rate_20steps')

	m.load_state_dict(torch.load('./model_dicts/linet_spiking_rate_20steps'))
	getAtol(m, dataloaders, device, encoding=True)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88
	"""

	# """
	# SNN, Rate, nSteps=50 , 10 epochs

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	best, last = pipeline(m, dataloaders, device, epochs=10, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/linet_spiking_rate_50steps_10epochs')

	# m.load_state_dict(torch.load('./model_dicts/linet_spiking_rate_50steps_10epochs'))
	getAtol(last, dataloaders, device, encoding=True)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0    0.12  5.4   66.96  99.88
	# """

if __name__ == "__main__":
	main()