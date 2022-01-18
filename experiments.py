import torch
from   torchvision import transforms

from data   import loadData, generateDataloaders, scaleDownData, nSteps
from data   import RateEncodeData, LatencyEncodeData, CopyEncodeLabels, OnOffChannels
from models import FC,  FCSpiking,  FCnoBias
from models import LCN, LCNSpiking, LCNnoBias, LCNSpikingHybrid, LCNChannelStack
from models import Model, LCNSpiking2
from train  import pipeline

from snntorch import surrogate


# cycle through a list of given models
def cycleThroughModels(device, models, nEpochs=1, dataType='', f=1, nSteps=nSteps, loadOpt=''):

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

	rate    = RateEncodeData(nSteps, 2, 0)
	latency = LatencyEncodeData(nSteps, 5, 0.01)
	# spikeLabels = CopyEncodeLabels(nSteps)
	dataSpiking = generateDataloaders(data, labels, xTransform=latency) 

	# Train
	for name, m in models.items():
		print(f'Training: {name}')

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

		best, last = pipeline(m, dataSpiking, device, epochs=nEpochs, lr=1e-3, weight_decay=0, encoding=encode, patience=7, atol=1e-2, saveOpt=name, loadOpt=opt)
		
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
		'ex1_
		LCNSpiking_delta_100epoch' : LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True)
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


	# TODO
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
		'ex16_LCNSpiking_spikes_normal_20epoch_k25_surrogate_lso' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		'ex16_LCNSpikingHybrid_spiking_normal_20epoch_k25_L1_surrogate_lso': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		'ex16_LCNSpikingHybrid_spiking_normal_20epoch_k25_L2_surrogate_lso': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		# 'ex16_LCNSpikingHybrid_spiking_normal_20epoch_k25_L3_surrogate_lso': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
		# 'ex16_LCNSpikingHybrid_spiking_normal_20epoch_k25_L4_surrogate_lso': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
	}

	cycleThroughModels(device, models, nEpochs=20)

	# models = {
	# 	'ex16_LCNSpiking_spikes_delta_20epoch_k25_surrogate_lso' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
	# 	'ex16_LCNSpikingHybrid_spiking_delta_20epoch_k25_L1_surrogate_lso': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
	# 	'ex16_LCNSpikingHybrid_spiking_delta_20epoch_k25_L2_surrogate_lso': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
	# 	'ex16_LCNSpikingHybrid_spiking_delta_20epoch_k25_L3_surrogate_lso': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
	# 	'ex16_LCNSpikingHybrid_spiking_delta_20epoch_k25_L4_surrogate_lso': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=lso),
	# }
	# cycleThroughModels(device, models, nEpochs=20, dataType='Delta')
	
	sigmoid = surrogate.sigmoid()

	models = {
		'ex16_LCNSpiking_spikes_normal_20epoch_k25_surrogate_sigmoid' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		'ex16_LCNSpikingHybrid_spiking_normal_20epoch_k25_L1_surrogate_sigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		'ex16_LCNSpikingHybrid_spiking_normal_20epoch_k25_L2_surrogate_sigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		# 'ex16_LCNSpikingHybrid_spiking_normal_20epoch_k25_L3_surrogate_sigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
		# 'ex16_LCNSpikingHybrid_spiking_normal_20epoch_k25_L4_surrogate_sigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	}

	cycleThroughModels(device, models, nEpochs=20)

	# models = {
	# 	'ex16_LCNSpiking_spikes_delta_20epoch_k25_surrogate_sigmoid' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	# 	'ex16_LCNSpikingHybrid_spiking_delta_20epoch_k25_L1_surrogate_sigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	# 	'ex16_LCNSpikingHybrid_spiking_delta_20epoch_k25_L2_surrogate_sigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	# 	'ex16_LCNSpikingHybrid_spiking_delta_20epoch_k25_L3_surrogate_sigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	# 	'ex16_LCNSpikingHybrid_spiking_delta_20epoch_k25_L4_surrogate_sigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=sigmoid),
	# }
	# cycleThroughModels(device, models, nEpochs=20, dataType='Delta')
	"""
	# stopped at L2 for normal, didn't do delta for either surrogate - fastsigmoid is the best so far
	# *************************************************************************
	

	# Ex 17: Train Spiking Models for full 100epoch (continuation of ex13) ****

	fastSigmoid = surrogate.fast_sigmoid()

	opt = {
		'ex13_LCNSpiking_spikes_normal_100epoch_k25_surrogate_fastSigmoid':           'ex13_LCNSpiking_spikes_normal_40epoch_k25_surrogate_fastSigmoid',
		'ex13_LCNSpikingHybrid_spiking_normal_100epoch_k25_L1_surrogate_fastSigmoid': 'ex13_LCNSpikingHybrid_spiking_normal_40epoch_k25_L1_surrogate_fastSigmoid',
		'ex13_LCNSpikingHybrid_spiking_normal_100epoch_k25_L2_surrogate_fastSigmoid': 'ex13_LCNSpikingHybrid_spiking_normal_40epoch_k25_L2_surrogate_fastSigmoid',
		'ex13_LCNSpikingHybrid_spiking_normal_100epoch_k25_L3_surrogate_fastSigmoid': 'ex13_LCNSpikingHybrid_spiking_normal_40epoch_k25_L3_surrogate_fastSigmoid',
		'ex13_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid': 'ex13_LCNSpikingHybrid_spiking_normal_40epoch_k25_L4_surrogate_fastSigmoid'
	}

	models = {
		'ex13_LCNSpiking_spikes_normal_100epoch_k25_surrogate_fastSigmoid' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_normal_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_normal_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_normal_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}

	cycleThroughModels(device, models, nEpochs=40, loadOpt=opt)

	opt = {
		'ex13_LCNSpiking_spikes_delta_100epoch_k25_surrogate_fastSigmoid':           'ex13_LCNSpiking_spikes_delta_40epoch_k25_surrogate_fastSigmoid',
		'ex13_LCNSpikingHybrid_spiking_delta_100epoch_k25_L1_surrogate_fastSigmoid': 'ex13_LCNSpikingHybrid_spiking_delta_40epoch_k25_L1_surrogate_fastSigmoid',
		'ex13_LCNSpikingHybrid_spiking_delta_100epoch_k25_L2_surrogate_fastSigmoid': 'ex13_LCNSpikingHybrid_spiking_delta_40epoch_k25_L2_surrogate_fastSigmoid',
		'ex13_LCNSpikingHybrid_spiking_delta_100epoch_k25_L3_surrogate_fastSigmoid': 'ex13_LCNSpikingHybrid_spiking_delta_40epoch_k25_L3_surrogate_fastSigmoid',
		'ex13_LCNSpikingHybrid_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid': 'ex13_LCNSpikingHybrid_spiking_delta_40epoch_k25_L4_surrogate_fastSigmoid'
	}

	models = {
		'ex13_LCNSpiking_spikes_delta_100epoch_k25_surrogate_fastSigmoid' : LCNSpiking2(14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_delta_100epoch_k25_L1_surrogate_fastSigmoid': LCNSpikingHybrid(1, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_delta_100epoch_k25_L2_surrogate_fastSigmoid': LCNSpikingHybrid(2, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_delta_100epoch_k25_L3_surrogate_fastSigmoid': LCNSpikingHybrid(3, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
		'ex13_LCNSpikingHybrid_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid': LCNSpikingHybrid(4, 14400, 2, 25, 2, 5, 0, 1, True, spikeGrad=fastSigmoid),
	}
	cycleThroughModels(device, models, nEpochs=40, dataType='Delta', loadOpt=opt)

	# *************************************************************************
	

if __name__ == "__main__":
	main()

# TODO experiments.py
"""
	gain experiments?

	nSteps vary (ex 12 didn't work, need to globally change nSteps in data.py)
"""

# TODO models.py
"""
	larger models with more layers
	
	make models where there are varying numbers of spiking layers
		change front and back, spiking and non-spiking

	model that encodes 2 channels

	loss function looks at all timesteps? 
"""

# TODO: train.py
"""
	add loss for different timesteps
"""
