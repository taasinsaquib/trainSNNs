
import torch

import numpy as np

import bindsnet
from bindsnet.conversion       import ann_to_snn
from bindsnet.learning         import PostPre
from bindsnet.network.monitors import Monitor

from models import LCN, FC
from data   import loadData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convertToSNN():

	# *************************************************************************
	# Load Data and Trained Model
	# *************************************************************************

	sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir)

	"""
	modelDictPath = './model_dicts/LCN_normal_200epoch'
	ann = LCN(14400, 2, 15, 2, 5, True)
	"""
	modelDictPath = './model_dicts/FC_normal_200epoch'
	ann = FC()

	ann.load_state_dict(torch.load(modelDictPath, map_location=device))

	# ann.eval()
	ann.to(torch.float)
	ann.to(device)

	# *************************************************************************
	# Convert to SNN
	# *************************************************************************

	"""
	d = np.random.rand(1, 14400)
	d = torch.from_numpy(d).float()
	d = d.to(device)

	out = ann(d)
	print("IH", out)
	"""

	# data used to get max output of each weight?
	d = data
	d = torch.from_numpy(d).float()
	d = d.to(device)

	snn = ann_to_snn(ann, input_shape=(1, 14400), data=d)
	snn.save('./model_dicts/bindsnet_snn_FC_normal_200epoch')


def main():

	
	print(f'Device: {device}')

	snn = bindsnet.network.network.load('./model_dicts/bindsnet_snn_FC_normal_200epoch', map_location=device, learning=True)

	# add STDP learning rule
	"""
	for idx in snn.connections:
		c = snn.connections[idx]

		c.update_rule = PostPre			# TODO: compare with paper, do I need to implement my own rule?
		c.nu = (1e-4, 1e-2)				# TODO: figure out how to tune
	"""

	# *************************************************************************
	# Train SNN (from 2-stage paper)
	# *************************************************************************

	sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir)
	

	time = 20

	"""
	# add Monitors
	for l in snn.layers:
		if l != 'Input':
			snn.add_monitor(
				Monitor(snn.layers[l], state_vars=['s', 'v'], time=time, device=device), name=l
			)
	for c in snn.connections:
		# if isinstance(snn.connections[c], MaxPool2dConnection):
		snn.add_monitor(
			Monitor(snn.connections[c], state_vars=['firing_rates'], time=time, device=device), name=f'{c[0]}_{c[1]}_rates'
		)
	"""
	
	# TODO: training loop

	# inpts = {'Input': images[i].repeat(time, 1, 1, 1, 1)}
	# d = data[:1]
	# d = torch.from_numpy(d).float()
	# d = d.to(device)

	inpts = {'Input': torch.from_numpy(data[:1]).float()}
	snn.run(inputs=inpts , time=time)


	spikes = {
		l: SNN.monitors[l].get('s') for l in SNN.monitors if 's' in SNN.monitors[l].state_vars
	}
	voltages = {
		l: SNN.monitors[l].get('v') for l in SNN.monitors if 'v' in SNN.monitors[l].state_vars
	}
	firing_rates = {
		l: SNN.monitors[l].get('firing_rates').view(-1, time) for l in SNN.monitors if 'firing_rates' in SNN.monitors[l].state_vars
	}


if __name__ == "__main__":
	main()