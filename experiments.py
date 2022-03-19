import torch
from   torchvision import transforms

from data   import loadData, generateDataloaders, scaleDownData, nSteps
from data   import RateEncodeData, LatencyEncodeData, CopyEncodeLabels, OnOffChannels, CopyRedChannel
from models import FC,  FCSpiking,  FCnoBias
from models import LCN, LCNSpiking, LCNnoBias, LCNSpikingHybrid, LCNSpikingHybrid2, LCNSpikingHybrid3, LCNChannelStack
from models import Model, LCNSpiking2
from train  import pipeline

from snntorch import surrogate


# cycle through a list of given models
def cycleThroughModels(device, models, nEpochs=1, dataType='', f=1, nSteps=nSteps, loadOpt='', rateGain=None):

	"""
		device   - string, "cpu" or "cuda:0", etc.
		models   - dict, string -> nn Model
		dataType - string, "" or "delta"
		f        - float, scale down dataset if needed
		nSteps   - int, number of steps to encode the data into spikes
		loadOpt  - dict, model name -> opt name

	"""

	# Prep Data
	sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir, name=dataType)
	data, labels = scaleDownData(data, labels, f)
	dataNormal = generateDataloaders(data, labels)

	# rate    = RateEncodeData(nSteps, 2, 0)
	rate    = RateEncodeData(nSteps, 1, 0)
	# latency = LatencyEncodeData(nSteps, 5, 0.01)
	# spikeLabels = CopyEncodeLabels(nSteps)
	rgb = CopyRedChannel()

	rateRGB = transforms.Compose([rgb, rate])

	dataSpiking = generateDataloaders(data, labels, xTransform=rateRGB) 

	# Train
	for name, m in models.items():
		print(f'Training: {name}') 

		if rateGain != None:
			rate    = RateEncodeData(nSteps, rateGain[name], 0)
			dataSpiking = generateDataloaders(data, labels, xTransform=rate) 

		if "SpikingHybrid" in name:
			encode = False
		elif "Spiking" in name:
			encode = True
		else:
			encode = False

		if loadOpt != '':
			opt = loadOpt[name]
		else:
			opt = ''

		best, last = pipeline(m, dataSpiking, device, epochs=nEpochs, lr=1e-3, weight_decay=0, encoding=encode, patience=10, atol=1e-2, saveOpt=name, loadOpt=opt)
		
		torch.save(last.state_dict(), f'./model_dicts/{name}')


# *****************************************************************************
# main
# *****************************************************************************

def main():
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Ex 1: Set a baseline for normal and delta data for each model ***********
	"""
	models = {
		'ex1_FC_normal_100epoch':          FC(),
		'ex1_LCN_normal_100epoch':         LCN(14400, 2, 15, 2, 5, True),
		'ex1_LCNSpiking_normal_100epoch' : LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True)
	}
	cycleThroughModels(device, models, nEpochs=100)

	models.clear()
	models = {
		'ex1_FC_delta_100epoch':          FC(),
		'ex1_LCN_delta_100epoch':         LCN(14400, 2, 15, 2, 5, True),
		'ex1_LCNSpiking_delta_100epoch' : LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True)
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')

	# TODO: add _k15 to the names in the right spots
	"""
	# *************************************************************************


	# Ex 2: Try other values of K for LCN *************************************
	"""
	models = {
		'LCN_normal_100epoch_k20':  LCN(14400, 2, 20, 2, 5, True),
		'LCN_normal_100epoch_k25' : LCN(14400, 2, 25, 2, 5, True)
	}
	cycleThroughModels(device, models, nEpochs=100)

	models = {
		'LCN_delta_100epoch_k20':  LCN(14400, 2, 20, 2, 5, True),
		'LCN_delta_100epoch_k25' : LCN(14400, 2, 25, 2, 5, True)
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""

	# Results: use k=25 from now on
	# *************************************************************************


	# Ex 3: Try LCNChannelStack ***********************************************
	
	"""
	sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir, "Delta")

	onoff       = OnOffChannels(14400)
	dataloaders = generateDataloaders(data, labels, xTransform=onoff)

	# name='LCNChannelStack_delta_100epoch_k25'

	# m = LCNChannelStack(14400, 2, 25, 2, 5, True)
	# m.to(torch.float)
	# m.to(device)

	# best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=False, patience=7, atol=1e-2, saveOpt=name)
	
	# torch.save(last.state_dict(), f'./model_dicts/{name}')

	# Spiking, too much memory
	name='LCNChannelStackSpiking_delta_100epoch_k25'
	rate = RateEncodeData(nSteps, 2, 0)
	dataloaders = generateDataloaders(data, labels, xTransform=transforms.Compose([onoff, rate]))

	m = LCNChannelStack(14400, 2, 25, 2, 5, True, spiking=True)
	m.to(torch.float)
	m.to(device)

	best, last = pipeline(m, dataloaders, device, epochs=100, lr=1e-3, weight_decay=0, encoding=False, patience=7, atol=1e-2, saveOpt=name)
	
	torch.save(last.state_dict(), f'./model_dicts/{name}')
	"""

	# *************************************************************************


	# Ex 4: Try the hybrid models *********************************************
	# have to comment out neuron layers in LCNSpiking, and set encoding=False in cycleThroughModels()
	"""
	models = {
		'LCNSpikingHybrid_normal_100epoch_k25_L1': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True),
		'LCNSpikingHybrid_normal_100epoch_k25_L2': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True),
		'LCNSpikingHybrid_normal_100epoch_k25_L3': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True),
		'LCNSpikingHybrid_normal_100epoch_k25_L4': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True),
	}

	cycleThroughModels(device, models, nEpochs=100)

	models = {
		'LCNSpikingHybrid_delta_100epoch_k25_L1': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True),
		'LCNSpikingHybrid_delta_100epoch_k25_L2': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True),
		'LCNSpikingHybrid_delta_100epoch_k25_L3': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True),
		'LCNSpikingHybrid_delta_100epoch_k25_L4': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""
	# *************************************************************************


	# Ex 5: Try FCSpiking *****************************************************
	# Cut short, tye again later and decrease learning rate

	"""
	sigDir = "training_data/siggraph_data"
	rate = RateEncodeData(nSteps, 2, 0)

	# Normal
	data, labels = loadData(sigDir)
	normalData = generateDataloaders(data, labels, xTransform=rate)

	m = FCSpiking(0, 1)
	m.to(torch.float)
	m.to(device)

	name ='FCSpiking_normal_100epoch'
	best, last = pipeline(m, normalData, device, epochs=100, lr=1e-3, weight_decay=0, encoding=False, patience=7, atol=1e-2, saveOpt=name)
	torch.save(last.state_dict(), f'./model_dicts/{name}')

	# Delta
	data, labels = loadData(sigDir, 'Delta')
	deltaData = generateDataloaders(data, labels, xTransform=rate)

	m = FCSpiking(0, 1)
	m.to(torch.float)
	m.to(device)

	name ='FCSpiking_delta_100epoch'
	best, last = pipeline(m, deltaData, device, epochs=100, lr=1e-3, weight_decay=0, encoding=False, patience=7, atol=1e-2, saveOpt=name)
	torch.save(last.state_dict(), f'./model_dicts/{name}')
	"""

	# *************************************************************************


	# Ex 6: Train FCnoBias for conversion *************************************
	"""
	models = {
		'FCnoBias_normal_100epoch': FCnoBias(),
		'LCNnoBias_normal_100epoch': LCNnoBias(14400, 2, 25, 2, 5, True)
	}
	cycleThroughModels(device, models, nEpochs=100)

	# didn't run delta, FC above took 10 hours and neither had good enough performance
	models = {
		'FCnoBias_delta_100epoch': FCnoBias(),
		'LCNnoBias_delta_100epoch': LCNnoBias(14400, 2, 25, 2, 5, True)
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""
	# *************************************************************************


	# Ex 7: Surrogate Gradients for Hybrid ************************************
	# set encoding=False in cycleThroughModels()
	"""
	fastSigmoid = surrogate.fast_sigmoid()
	
	models = {
		'LCNSpikingHybrid_normal_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'LCNSpikingHybrid_normal_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100)

	models = {
		'LCNSpikingHybrid_delta_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'LCNSpikingHybrid_delta_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	 """
	
	# *************************************************************************


	# Ex 8: Train LCNSpiking with k = 25 ************************************
	"""
	models = {
		'LCNSpiking_normal_100epoch_k25' : LCNSpiking(14400, 2, 25, 2, 5, 0, 1, True)
	}
	cycleThroughModels(device, models, nEpochs=100)

	models.clear()
	models = {
		'LCNSpiking_delta_100epoch_k25' : LCNSpiking(14400, 2, 25, 2, 5, 0, 1, True)
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""
	# *************************************************************************

	# FIX MODEL #1

	# Ex 9: Train LCNSpiking with actual spikes lol ***************************
	"""
	fastSigmoid = surrogate.fast_sigmoid()

	models = {
		'LCNSpiking_spikes_normal_100epoch_k25' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True),
		'LCNSpiking_spikes_normal_100epoch_k25_surrogate_fastSigmoid' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid)
	}
	cycleThroughModels(device, models, nEpochs=20)

	models.clear()
	models = {
		'LCNSpiking_spikes_delta_100epoch_k25' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True),
		'LCNSpiking_spikes_delta_100epoch_k25_surrogate_fastSigmoid' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid)
	}
	cycleThroughModels(device, models, nEpochs=20, dataType='Delta')
	# """
	# results - use surrogate gradient
	# *************************************************************************


	# Ex 10: Try the hybrid models with actual spikes *************************
	# have to comment out neuron layers in LCNSpiking, and set encoding=False in cycleThroughModels()
	"""
	models = {
		'ex10_LCNSpikingHybrid_spiking_normal_20epoch_k25_L1': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True),
		'ex10_LCNSpikingHybrid_spiking_normal_20epoch_k25_L2': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True),
		'ex10_LCNSpikingHybrid_spiking_normal_20epoch_k25_L3': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True),
		'ex10_LCNSpikingHybrid_spiking_normal_20epoch_k25_L4': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True),
	}

	cycleThroughModels(device, models, nEpochs=20)

	# didn't run, do later after figuring out L3 and L4 performance
	# models = {
	# 	'LCNSpikingHybrid_spiking_delta_100epoch_k25_L1': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True),
	# 	'LCNSpikingHybrid_spiking_delta_100epoch_k25_L2': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True),
	# 	'LCNSpikingHybrid_spiking_delta_100epoch_k25_L3': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True),
	# 	'LCNSpikingHybrid_spiking_delta_100epoch_k25_L4': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True),
	# }
	# cycleThroughModels(device, models, nEpochs=20, dataType='Delta')
	"""
	# *************************************************************************


	# Ex 11: Train L3 and L4 for more epochs **********************************

	"""
	m3 = LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True)
	m3.load_state_dict(torch.load(f'./model_dicts/LCNSpikingHybrid_spiking_normal_100epoch_k25_L3', map_location=device))

	models = {
		'LCNSpikingHybrid_spiking_normal_150epoch_k25_L3': m3,
	}

	cycleThroughModels(device, models, nEpochs=50)
	"""
	# results - train at least 100 epochs
	# *************************************************************************


	# Ex 12: Vary Num steps for L3 and L4 *************************************
	"""
	models = {
		'ex12_LCNSpikingHybrid_spiking_normal_100epoch_k25_L3_nSteps50': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True),
		# 'LCNSpikingHybrid_spiking_normal_100epoch_k25_L4': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True),
	}
	cycleThroughModels(device, models, nEpochs=100, nSteps=50)

	# didn't run 100 steps
	models = {
		'LCNSpikingHybrid_spiking_normal_100epoch_k25_L3_nSteps100': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True),
		# 'LCNSpikingHybrid_spiking_normal_100epoch_k25_L4': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True),
	}
	cycleThroughModels(device, models, nEpochs=100, nSteps=100)
	"""

	# 50 didn't seem to to anything, doubt 100 willl
	# *************************************************************************


	# Ex 13: 100 epochs, delta, for LCN Spiking Spiking, L1, L2 ***************
	"""
	fastSigmoid = surrogate.fast_sigmoid()

	models = {
		'ex13_LCNSpiking_spikes_normal_40epoch_k25_surrogate_fastSigmoid' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_normal_40epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_normal_40epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_normal_40epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_normal_40epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=40)

	models = {
		'ex13_LCNSpiking_spikes_normal_40epoch_k25_surrogate_fastSigmoid' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),					# forgot to change this name to delta ...
		'ex13_LCNSpikingHybrid_spiking_delta_40epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_delta_40epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_delta_40epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_delta_40epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=40, dataType='Delta')
	"""
	# beats LCN k=15
	# *************************************************************************


	# Ex 14: lower learning rate for LCNSpiking2 ******************************
	"""	
	# changed lr in cycleThroughModels to 1e-4

	fastSigmoid = surrogate.fast_sigmoid()

	models = {
		'ex14_LCNSpiking_spikes_normal_40epoch_k25_surrogate_fastSigmoid_lr1e-4' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=40)

	models = {
		'ex14_LCNSpiking_spikes_normal_40epoch_k25_surrogate_fastSigmoid_lr1e-4' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=40, dataType='Delta')
	"""
	# 1e-5 could work, but in general these models wont work (can't get any training correct)
	# *************************************************************************
	

	# Ex 15: Try Latency on normal data ***************************************
	"""
	fastSigmoid = surrogate.fast_sigmoid()

	models = {
		'ex15_LCNSpiking_spikes_normal_latency_20epoch_k25_surrogate_fastSigmoid' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex15_LCNSpikingHybrid_spiking_normal_latency_20epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex15_LCNSpikingHybrid_spiking_normal_latency_20epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex15_LCNSpikingHybrid_spiking_normal_latency_20epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex15_LCNSpikingHybrid_spiking_normal_latency_20epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=20)
	"""
	# latency is terrible
	# *************************************************************************


	# Ex 16: Try other two surrogates *****************************************
	"""
	lso     = surrogate.LSO()

	models = {
		'ex16_LCNSpikingHybrid_spiking_normal_10epoch_k25_L1_surrogate_lso': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		# 'ex16_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_surrogate_lso': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		# 'ex16_LCNSpikingHybrid_spiking_normal_10epoch_k25_L3_surrogate_lso': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		'ex16_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_surrogate_lso': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		'ex16_LCNSpiking_spikes_normal_10epoch_k25_surrogate_lso':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
	}

	cycleThroughModels(device, models, nEpochs=10)

	models = {
		'ex16_LCNSpikingHybrid_spiking_delta_10epoch_k25_L1_surrogate_lso': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		# 'ex16_LCNSpikingHybrid_spiking_delta_10epoch_k25_L2_surrogate_lso': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		# 'ex16_LCNSpikingHybrid_spiking_delta_10epoch_k25_L3_surrogate_lso': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		'ex16_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_surrogate_lso': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		'ex16_LCNSpiking_spikes_delta_10epoch_k25_surrogate_lso':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
	}
	cycleThroughModels(device, models, nEpochs=10, dataType='Delta')


	sigmoid = surrogate.sigmoid()

	models = {
		'ex16_LCNSpikingHybrid_spiking_normal_10epoch_k25_L1_surrogate_sigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		# 'ex16_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_surrogate_sigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		# 'ex16_LCNSpikingHybrid_spiking_normal_10epoch_k25_L3_surrogate_sigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		'ex16_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_surrogate_sigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		'ex16_LCNSpiking_spikes_normal_10epoch_k25_surrogate_sigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	}

	cycleThroughModels(device, models, nEpochs=10)

	models = {
		'ex16_LCNSpikingHybrid_spiking_delta_10epoch_k25_L1_surrogate_sigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		# 'ex16_LCNSpikingHybrid_spiking_delta_10epoch_k25_L2_surrogate_sigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		# 'ex16_LCNSpikingHybrid_spiking_delta_10epoch_k25_L3_surrogate_sigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		'ex16_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_surrogate_sigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		'ex16_LCNSpiking_spikes_delta_10epoch_k25_surrogate_sigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	}
	cycleThroughModels(device, models, nEpochs=10, dataType='Delta')
	"""
	# only L1 good with lso, about the same with sigmoid
	# just stick to fast sigmoid
	# *************************************************************************

	fastSigmoid = surrogate.fast_sigmoid()

	# Ex 17: Train Spiking Models for full 100epoch (continuation of ex13) ****
	"""
	# didn't get same results as 13, look at it again tmrrw - latency encoding was used accidentally on ex16 and 17
	

	models = {
		'ex17_LCNSpikingHybrid_spiking_normal_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex17_LCNSpikingHybrid_spiking_normal_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex17_LCNSpikingHybrid_spiking_normal_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex17_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex17_LCNSpiking_spikes_normal_100epoch_k25_surrogate_fastSigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=100)

	models = {
		'ex17_LCNSpikingHybrid_spiking_delta_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex17_LCNSpikingHybrid_spiking_delta_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex17_LCNSpikingHybrid_spiking_delta_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex17_LCNSpikingHybrid_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex17_LCNSpiking_spikes_delta_100epoch_k25_surrogate_fastSigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""
	# *************************************************************************


	# Ex 18: Vary Num steps (same as ex12)  ***********************************
	# change nSteps in data.py

	"""
	models = {
		'ex18_LCNSpikingHybrid_spiking_normal_100epoch_k25_L1_nSteps50': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		'ex18_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_nSteps50': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		'ex18_LCNSpiking_spiking_normal_100epoch_k25_nSteps50':          LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	}
	cycleThroughModels(device, models, nEpochs=10)

	# didn't run 100 steps
	# models = {
	# 	'ex18_LCNSpikingHybrid_spiking_delta_100epoch_k25_L1_nSteps100': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	# 	'ex18_LCNSpikingHybrid_spiking_delta_100epoch_k25_L4_nSteps100': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	# 	'ex18_LCNSpiking_spiking_delta_100epoch_k25_nSteps100':          LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	# }
	# cycleThroughModels(device, models, nEpochs=10)
	"""

	# more steps seems to help LCNSpiking, but not significantly. Maybe it needs an order of magnitude more
	# *************************************************************************


	# Ex 19: LCNSpiking with diff loss calculation  ***************************

	"""
	sigDir = "training_data/siggraph_data"
	rate = RateEncodeData(nSteps, 2, 0)
	spikeLabels = CopyEncodeLabels(nSteps)

	# Normal
	data, labels = loadData(sigDir)
	data, labels = scaleDownData(data, labels, 0.01)
	normalData = generateDataloaders(data, labels, xTransform=rate)

	m = LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid)
	m.to(torch.float)
	m.to(device)

	name ='ex19_LCNSpiking_spiking_normal_lossCalc'
	best, last = pipeline(m, normalData, device, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2, saveOpt=name)
	# torch.save(last.state_dict(), f'./model_dicts/{name}')

	# Delta
	# data, labels = loadData(sigDir, 'Delta')
	# deltaData = generateDataloaders(data, labels, xTransform=rate)

	# m = FCSpiking(0, 1)
	# m.to(torch.float)
	# m.to(device)

	# name ='FCSpiking_delta_100epoch'
	# best, last = pipeline(m, deltaData, device, epochs=100, lr=1e-3, weight_decay=0, encoding=False, patience=7, atol=1e-2, saveOpt=name)
	# torch.save(last.state_dict(), f'./model_dicts/{name}')
	"""

	# *************************************************************************


	# Ex 20: Sweep Gain  ******************************************************
	"""
	models = {
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain0.5': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain1': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain2': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain5': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain10': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain50': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain100': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}


	gains = {
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain0.5': 0.5,
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain1': 1,
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain2': 2,
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain5': 5,
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain10': 10,
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain50': 50,
		'ex20_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_gain100': 100
	}

	cycleThroughModels(device, models, nEpochs=10, rateGain=gains)
	"""

	# TODO: Delta
	"""
	models = {
		'ex20_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_gain5': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex20_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_gain10': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex20_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_gain50': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex20_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_gain100': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	gains = {
		'ex20_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_gain5': 5,
		'ex20_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_gain10': 10,
		'ex20_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_gain50': 50,
		'ex20_LCNSpikingHybrid_spiking_delta_10epoch_k25_L4_gain100': 100
	}

	cycleThroughModels(device, models, nEpochs=10, dataType='Delta', rateGain=gains)
	"""
	# for normal data, 50 seems to be a good middle ground
	# for delta, ...
	# *************************************************************************


	# Ex 21: Sweep Alpha  ******************************************************
	"""
	models = {
		'ex21_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_alpha0':    LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0,    1, True, spikeGrad=fastSigmoid),
		'ex21_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_alpha0.25': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0.25, 1, True, spikeGrad=fastSigmoid),
		'ex21_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_alpha0.5':  LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0.5,  1, True, spikeGrad=fastSigmoid),
		'ex21_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_alpha1':    LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 1,    1, True, spikeGrad=fastSigmoid)
	}

	cycleThroughModels(device, models, nEpochs=10)
	"""
	# alpha doesn't seem to help
	# *************************************************************************

	# FIX MODEL #2

	# Ex 22: Correctly Assign Membranes, run L1 through Spiking for normal  ***
	# Same as ex17

	"""
	models = {
		'ex22_LCNSpikingHybrid_spiking_normal_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex22_LCNSpikingHybrid_spiking_normal_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex22_LCNSpikingHybrid_spiking_normal_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex22_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex22_LCNSpiking_spikes_normal_100epoch_k25_surrogate_fastSigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=100)
	
	# change back to 16 batch size
	models = {
		'ex22_LCNSpikingHybrid_spiking_delta_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex22_LCNSpikingHybrid_spiking_delta_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex22_LCNSpikingHybrid_spiking_delta_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex22_LCNSpikingHybrid_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	# 	'ex22_LCNSpiking_spikes_delta_100epoch_k25_surrogate_fastSigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	# Run normal after
	"""
	# *************************************************************************


	# Ex 23: Leaky Neurons (same as 22) ***************************************
	# Change model used in Hybrid to _Leaky
	"""
	models = {
		'ex23_LCNSpikingHybrid_Leaky_spiking_normal_10epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex23_LCNSpikingHybrid_Leaky_spiking_normal_10epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex23_LCNSpikingHybrid_Leaky_spiking_normal_10epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex23_LCNSpikingHybrid_Leaky_spiking_normal_10epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex23_LCNSpiking_spiking_Leakys_normal_10epoch_k25_surrogate_fastSigmoid':         LCNSpiking2_Leaky(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=10)

	# models = {
	# 	'ex22_LCNSpikingHybrid_Leaky_spiking_delta_10epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	# 	'ex22_LCNSpikingHybrid_Leaky_spiking_delta_10epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	# 	'ex22_LCNSpikingHybrid_Leaky_spiking_delta_10epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	# 	'ex22_LCNSpikingHybrid_Leaky_spiking_delta_10epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	# 	'ex22_LCNSpiking_Leaky_spikes_delta_10epoch_k25_surrogate_fastSigmoid':           LCNSpiking2_Leaky(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	# }
	# cycleThroughModels(device, models, nEpochs=10, dataType='Delta')
	"""
	# not as good as synaptic, only ran normal up to L3, no delta
	# *************************************************************************	


	# Ex 24: Leaky Neurons (same as 22) ***************************************
	# 22 but with synaptic inputs correctly placed ugh...
	# accidentally just ran 22, with 10's changed to 100's

	# *************************************************************************	

	# TODO: Test gain and alpha on the best models of 22
	# Ex 25: Sweep Alpha pt 2 *************************************************
	"""
	models = {
		'ex25_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_alpha1':    LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 1,    1, True, spikeGrad=fastSigmoid),
		'ex25_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_alpha0.75': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0.75, 1, True, spikeGrad=fastSigmoid),
		'ex25_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_alpha0.5':  LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0.5,  1, True, spikeGrad=fastSigmoid),
		'ex25_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_alpha0.25': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0.25, 1, True, spikeGrad=fastSigmoid),
		'ex25_LCNSpikingHybrid_spiking_normal_10epoch_k25_L4_alpha0':    LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0,    1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=10)
	"""
	# 0 alpha is the best
	# *************************************************************************

	# Ex 26-27: Sweep Gain pt 2 ***********************************************
	"""
	models = {
		'ex26_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain5': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex26_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain10': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex26_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain50': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex26_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain100': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	gains = {
		'ex26_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain5': 5,
		'ex26_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain10': 10,
		'ex26_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain50': 50,
		'ex26_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain100': 100
	}

	cycleThroughModels(device, models, nEpochs=10, rateGain=gains)
	
	# doesn't really help


	# synaptic[]i kept to initial value, not updated in forward
	models = {
		# 'ex27_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain5': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex27_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain10': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex27_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain50': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex27_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain100': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	gains = {
		# 'ex27_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain5': 5,
		# 'ex27_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain10': 10,
		# 'ex27_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain50': 50,
		# 'ex27_LCNSpikingHybrid_spiking_normal_10epoch_k25_L2_gain100': 100
	}

	cycleThroughModels(device, models, nEpochs=10, rateGain=gains)
	"""
	# definitely keep synaptic[i] not updated
	# for normal data, 50 seems to be a good middle ground (100 is similar but slightly higher val loss)
	# *************************************************************************


	# Ex 28: 100ep, no synaptic, 50 gain  ***
	# Same as ex17

	"""
	models = {
		'ex28_LCNSpikingHybrid_spiking_normal_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex28_LCNSpikingHybrid_spiking_normal_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex28_LCNSpikingHybrid_spiking_normal_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex28_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex28_LCNSpiking_spikes_normal_100epoch_k25_surrogate_fastSigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=100)
	
	# change back to 16 batch size
	models = {
		'ex28_LCNSpikingHybrid_spiking_delta_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex28_LCNSpikingHybrid_spiking_delta_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex28_LCNSpikingHybrid_spiking_delta_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex28_LCNSpikingHybrid_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex28_LCNSpiking_spikes_delta_100epoch_k25_surrogate_fastSigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	# 5 layer SNN sucks
	"""
	# *************************************************************************


	# Try Latency One More Time ***********************************************

	# models = {
		# 'ex29_LCNSpikingHybrid_normal_100epoch_L1_latency': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	# }
	# cycleThroughModels(device, models, nEpochs=100)
	# converges to loss of about 0.2
	# *************************************************************************


	# Ex 30: Try Scaling Fast Sigmoid *****************************************
	# set gain to 2
	"""
	models = {
		# 'ex30_LCNSpikingHybrid_spiking_normal_20epoch_k25_L1_surrogate_fastSigmoid1':   LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=1)),
		# 'ex30_LCNSpikingHybrid_spiking_normal_20epoch_k25_L1_surrogate_fastSigmoid10':  LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=10)),
		# 'ex30_LCNSpikingHybrid_spiking_normal_20epoch_k25_L1_surrogate_fastSigmoid100': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=100)),
		'ex30_LCNSpikingHybrid_spiking_normal_20epoch_k25_L4_surrogate_fastSigmoid10':  LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=10)),
	}

	cycleThroughModels(device, models, nEpochs=20)
	"""

	"""
	models = {
		'ex30_LCNSpikingHybrid_spiking_normal_100epoch_k25_L1_surrogate_fastSigmoid10':  LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=10)),
		'ex30_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid10':  LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=10)),
	}

	cycleThroughModels(device, models, nEpochs=100)

	models = {
		'ex30_LCNSpikingHybrid_spiking_delta_100epoch_k25_L1_surrogate_fastSigmoid10':  LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=10)),
		'ex30_LCNSpikingHybrid_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid10':  LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=10)),
	}

	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""
	# better learning at first, converges to about the same
	# L4 normal is about 1% higher training

	# *************************************************************************

	# Ex 31: Thresholds to One ************************************************
	"""
	# torch.rand -> ones
	models = {
		'ex31_LCNSpikingHybrid_spiking_normal_20epoch_k25_L1_surrogate_fastSigmoid_thresholdOnes': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=25)),
		'ex31_LCNSpikingHybrid_spiking_normal_20epoch_k25_L4_surrogate_fastSigmoid_thresholdOnes': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=25)),
	}

	cycleThroughModels(device, models, nEpochs=20)
	# good results for L1 (beats ex22) but not L4 (does worse than delta ex22)


	# don't train thresholds
	# set train tresholds = False

	models = {
	# 	# 'ex31_LCNSpikingHybrid_spiking_normal_20epoch_k25_L1_surrogate_fastSigmoid_thresholdOnes_noTrain': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=25)),
		'ex31_LCNSpikingHybrid_spiking_normal_20epoch_k25_L4_surrogate_fastSigmoid_thresholdOnes_noTrain': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=25)),
	}

	cycleThroughModels(device, models, nEpochs=20)
	# didn't run L1, L4 exact same as just initializing to ones
	"""
	# *************************************************************************


	# Ex 32: More Epochs for L4 ***********************************************

	"""
	m4 = LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=25))
	m4.load_state_dict(torch.load(f'./model_dicts/ex22_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid', map_location=device))

	models = {
		'ex32_LCNSpikingHybrid_spiking_normal_200epoch_k25_L4_surrogate_fastSigmoid': m4
	}

	opt = {
		'ex32_LCNSpikingHybrid_spiking_normal_200epoch_k25_L4_surrogate_fastSigmoid': 'ex22_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid'
	}

	cycleThroughModels(device, models, nEpochs=100, loadOpt=opt)
	
	# doesn't help lol
	"""

	"""
	models = {
		'ex32_LCNSpikingHybrid_spiking_normal_500epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=surrogate.fast_sigmoid(slope=25))
	}
	cycleThroughModels(device, models, nEpochs=500)
	# loss was stuck didn't save lrsheduler un-comment out
	"""
	# *************************************************************************


	# Ex 33 LR Schedule Decay *************************************************

	"""
	models = {
		# 'ex33_LCNSpikingHybrid_spiking_normal_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex33_LCNSpikingHybrid_spiking_normal_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex33_LCNSpikingHybrid_spiking_normal_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex33_LCNSpikingHybrid_spiking_normal_500epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex33_LCNSpiking_spikes_normal_100epoch_k25_surrogate_fastSigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=100)

	models = {
		# 'ex33_LCNSpikingHybrid_spiking_delta_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex33_LCNSpikingHybrid_spiking_delta_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex33_LCNSpikingHybrid_spiking_delta_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex33_LCNSpikingHybrid_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex33_LCNSpiking_spikes_delta_100epoch_k25_surrogate_fastSigmoid':           LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')

	# tried gain 1, stick with gain 2
	# full spiking still doesn't converge
	"""

	"""
	models = {
		'ex33_LCN_normal_100epoch':         LCN(14400, 2, 25, 2, 5, True),
	}
	cycleThroughModels(device, models, nEpochs=100)

	models.clear()
	models = {
		'ex33_LCN_delta_100epoch':         LCN(14400, 2, 25, 2, 5, True),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""

	# *************************************************************************


	# Ex 34: RGB LCN **********************************************************
	"""
	models = {
		# 'ex34_LCN_normal_20epoch_k15_RGB_f5': LCN(43200, 2, 15, 5, 5, True),
		# 'ex34_LCN_normal_20epoch_k20_RGB_f5': LCN(43200, 2, 20, 5, 5, True),
		'ex34_LCN_normal_100epoch_k25': LCN(43200, 2, 25, 5, 5, True),
	}
	cycleThroughModels(device, models, nEpochs=100)

	models = {
		# 'ex34_LCN_delta_20epoch_k15_RGB_f5': LCN(43200, 2, 15, 5, 5, True),
		# 'ex34_LCN_delta_20epoch_k20_RGB_f5': LCN(43200, 2, 20, 5, 5, True),
		'ex34_LCN_delta_100epoch_k25': LCN(43200, 2, 25, 5, 5, True),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""
	# *************************************************************************


	# Ex 35: RGB LCNSpiking2 **************************************************
	"""
	models = {
		'ex35_LCNSpikingHybrid_spiking_normal_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex35_LCNSpikingHybrid_spiking_normal_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex35_LCNSpikingHybrid_spiking_normal_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex35_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex35_LCNSpiking_spikes_normal_100epoch_k25_surrogate_fastSigmoid':          LCNSpiking2(43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=100)

	models = {
		'ex35_LCNSpikingHybrid_spiking_delta_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex35_LCNSpikingHybrid_spiking_delta_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex35_LCNSpikingHybrid_spiking_delta_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex35_LCNSpikingHybrid_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex35_LCNSpiking_spikes_delta_10epoch_k25_surrogate_fastSigmoid':           LCNSpiking2(43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""
	# *************************************************************************


	# Ex 36: No ANN in Hybrid *************************************************
	"""
	models = {
		# 'ex36_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid2(1, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex36_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid2(2, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex36_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid2(3, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex36_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid2(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex36_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L5_surrogate_fastSigmoid': LCNSpikingHybrid2(5, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100)

	models = {
		# 'ex36_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid2(1, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex36_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid2(2, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex36_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid2(3, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex36_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid2(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		# 'ex36_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L5_surrogate_fastSigmoid': LCNSpikingHybrid2(5, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')

	# lr scheduling
	models = {
		'ex36_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid_LR': LCNSpikingHybrid2(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100)

	# lr scheduling
	models = {
		'ex36_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid_LR': LCNSpikingHybrid2(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	"""
	# *************************************************************************

	# Ex 37: Linear Transform at End ******************************************
	# """
	models = {
		# 'ex37_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L3_surrogate_fastSigmoid_LR': LCNSpikingHybrid2(3, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex37_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid_gain1': LCNSpikingHybrid2(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100)

	models = {
		# 'ex37_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L3_surrogate_fastSigmoid_LR': LCNSpikingHybrid2(3, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex37_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid_gain1': LCNSpikingHybrid2(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=100, dataType='Delta')

	# # lr scheduling
	# models = {
	# 	'ex36_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid_LR': LCNSpikingHybrid2(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	# }
	# cycleThroughModels(device, models, nEpochs=100)

	# # lr scheduling
	# models = {
	# 	'ex36_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid_LR': LCNSpikingHybrid2(4, 43200, 2, 25, 5, 5, 0, 1, True, spikeGrad=fastSigmoid),
	# }
	# cycleThroughModels(device, models, nEpochs=100, dataType='Delta')
	# """
	# *************************************************************************

if __name__ == "__main__":
	main()

# TODO experiments.py
"""
	nSteps vary (ex 12 didn't work, need to globally change nSteps in data.py)
"""

# TODO models.py
"""
	larger models with more layers
	
	add linear layers after LCNSpiking

	make models where there are varying numbers of spiking layers
		change front and back, spiking and non-spiking

"""

# TODO: train.py
"""
	add loss for different timesteps
"""
