import torch

from snntorch import surrogate

from models import LCN, LCNSpiking, LCNSpikingHybrid, ShallowNN, LCNChannelStack
from data   import CopyRedChannel, OffSpikes, RateEncodeData, LatencyEncodeData, CopyEncodeLabels
from data   import loadData, generateDataloaders, nSteps, subtractMean, normalization, scaleDownData
from train  import pipeline, getAtol, testModel

def normalLCN(device, f=1):

	# 100 epochs, batchSize=64
	sigDir = "training_data/siggraph_data"

	# NORMAL ONV **********************************************

	# """
	data, labels = loadData(sigDir)
	dataloaders = generateDataloaders(data, labels) 

	model = LCN(14400, 2, 15, 2, 5, True)
	best, last = pipeline(model, dataloaders, device, epochs=200, lr=1e-3, weight_decay=0, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/LCN_normal_200epoch')

	# testModel(model, './model_dicts/linet_normal_100epoch', dataloaders, device)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1   0.5
	#       0    0.04 2.23  61.0  99.4   100
	# """

	# DELTA ONV ***********************************************

	"""
	data, labels = loadData(sigDir, 'Delta')
	data, labels = scaleDownData(data, labels, f)
	dataloaders = generateDataloaders(data, labels)

	model = LCN(14400, 2, 15, 2, 5, True)

	best, last = pipeline(model,dataloaders, device, epochs=200, lr=1e-3, weight_decay=0, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/LCN_delta_200epoch')

	# testModel(model, './model_dicts/linet_deltaData_100epoch', dataloaders, device)
	"""

def rateGain(device, f=1):

	sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir, 'Delta')
	data, labels = scaleDownData(data, labels, f)
	spikeLabels = CopyEncodeLabels(nSteps)

	# 1.5	
	rate    = RateEncodeData(nSteps, 1.5, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/SNN_newLoss_100epoch_200pt_rate_gain1.5')

	# 2
	rate    = RateEncodeData(nSteps, 2, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/SNN_newLoss_100epoch_200pt_rate_gain2')

	# 2.5
	rate    = RateEncodeData(nSteps, 2.5, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/SNN_newLoss_100epoch_200pt_rate_gain2.5')


def sweepSurrogates(device, f=1):

	sigDir = "training_data/siggraph_data"
	rate    = RateEncodeData(nSteps, 2, 0)
	spikeLabels = CopyEncodeLabels(nSteps)

	data, labels = loadData(sigDir, 'Delta')
	data, labels = scaleDownData(data, labels, f)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	sigmoid     = surrogate.sigmoid()
	fastSigmoid = surrogate.fast_sigmoid()
	# lso       = surrogate.LeakySpikeOperator()
	
	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True, spikeGrad=sigmoid)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/SNN_newLoss_100epoch_200pt_rate_gain1.5_sigmoid')

	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True, spikeGrad=fastSigmoid)
	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/SNN_newLoss_100epoch_200pt_rate_gain1.5_fastSigmoid')


def sweepAlpha(device, f=1):
	sigDir = "training_data/siggraph_data"
	rate    = RateEncodeData(nSteps, 2, 0)
	spikeLabels = CopyEncodeLabels(nSteps)

	data, labels = loadData(sigDir, "Delta")
	data, labels = scaleDownData(data, labels, f)
	dataloaders = generateDataloaders(data, labels, xTransform=rate, yTransform=spikeLabels)

	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_sweepAlpha_0')

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.25, 1, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_sweepAlpha_0.25')

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.5, 1, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_sweepAlpha_0.5')

	m = LCNSpiking(14400, 2, 15, 2, 5, 0.75, 1, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_sweepAlpha_0.75')

	m = LCNSpiking(14400, 2, 15, 2, 5, 1, 1, True)
	best, last = pipeline(m, dataloaders, device, epochs=30, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), './model_dicts/snn_sweepAlpha_1')


def paramSweep(device, f=1):
	# Delta Data (copy for normal data)
	# LCN
	normalLCN(device, f)
	#

	# SNN
	## First Order (others: k neighbors, inhibition)
	rateGain(device, f)				# SNN Rate (Sweep 1.5 - 2.5)
	sweepSurrogates(device, f)      # Surrogate Gradients (2/3 types)
	##
	## Second Order
	# SNN Rate (Sweep Alpha w/ best beta and surrogate)
	sweepAlpha(device, f)
	##
	#

def main():
	pass


if __name__ == "__main__":
	main()