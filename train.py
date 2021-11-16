import math
import h5py
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import Dataset, TensorDataset, random_split
from   torch.utils.tensorboard  import SummaryWriter
from   torch.optim.lr_scheduler import ReduceLROnPlateau

from   torchvision import transforms

# ! pip install snntorch --quiet
import snntorch as snn
from   snntorch import spikegen
from   snntorch import backprop
from   snntorch import surrogate
import snntorch.spikeplot as splt

# ! pip install livelossplot --quiet
from livelossplot import PlotLosses


# *******************************************
# Constants

D2R = np.pi/180
R2D = 180/np.pi

seed = 528
np.random.seed(seed)
torch.manual_seed(seed)

batchSize = 16
nSteps = 20


# *******************************************
# Data Functions

def loadData(dir, name=''):
	data   = np.load(f'./{dir}/data{name}.npy',   mmap_mode='r+')
	labels = np.load(f'./{dir}/labels{name}.npy', mmap_mode='r+')  # TODO: turn into degrees or keep radians?

	print(data.shape, labels.shape)

	return data, labels

def createDeltaOnv(data, labels, dir):
	n = data.shape[0]
	
	cur = data[1:]
	prev = data[:n-1]

	delta = cur - prev
	np.save(f'{dir}/dataDelta.npy', delta)
	np.save(f'{dir}/labelsDelta.npy', labels[1:])

	return cur - prev, labels[1:]


def scaleDownData(data, labels, factor=1):
	n = int(data.shape[0] * factor)

	return data[:n], labels[:n]


def generateDataloaders(data, labels, xTransform=None, yTransform=None, numWorkers=1):

	X_data_tensor = torch.from_numpy(data).float()
	y_data_tensor = torch.from_numpy(labels).float()
	init_dataset = TensorDataset(X_data_tensor, y_data_tensor)

	# split train, val, test
	lengths = np.array([0.8, 0.1, 0.1])
	lengths *= int(len(init_dataset))
	lengths = np.rint(lengths)

	diff = data.shape[0] - np.sum(lengths)
	lengths[0] += diff
	print(f'Splits: {lengths}, n: {data.shape[0]}, adjustment: {diff}')

	lengths = np.asarray(lengths, dtype=np.int32)

	subset_train, subset_val, subset_test = random_split(init_dataset, lengths, generator=torch.Generator().manual_seed(seed)) 

	train_data = ONVData(
			subset_train, xTransform=xTransform, yTransform=yTransform)

	val_data = ONVData(
			subset_val,   xTransform=xTransform, yTransform=yTransform)

	test_data = ONVData(
			subset_test,  xTransform=xTransform, yTransform=yTransform)

	dataloaders = {
			'train': torch.utils.data.DataLoader(train_data, batch_size=batchSize, shuffle=True, num_workers=numWorkers),
			'val':   torch.utils.data.DataLoader(val_data,   batch_size=batchSize, shuffle=True, num_workers=numWorkers),
			'test':  torch.utils.data.DataLoader(test_data,  batch_size=batchSize, shuffle=True, num_workers=numWorkers),
	}

	return dataloaders


# *******************************************
# Dataloaders

	#  TODO: pass filepath when dataset too large for memory
class ONVData(Dataset): 
		def __init__(self, subset, xTransform=None, yTransform=None):
				self.subset = subset
				self.xTransform = xTransform
				self.yTransform = yTransform

		def __len__(self):
				return len(self.subset)

		def __getitem__(self, index):
				x, y = self.subset[index]

				if self.xTransform:
					x = self.xTransform(x)

				if self.yTransform:
					y = self.yTransform(y)

				return x, y

# copy ONV 3 times because RGB is all the same (white ball)
class CopyRedChannel():
	def __call__(self, x):
		x = np.tile(x, 3)
		return x

# if binary changes in color {-1, 0, 1}, assign probabilities to each of the 3 possibilities
class ChangeProbabilities():
	def __init__(self, neg, pos):
		self.neg = neg
		self.pos = pos
	
	def __call__(self, x):
		x[x == -1] = self.neg
		x[x == 1]  = self.pos
		return x

# turn negative magnitude change into positive delta for encoding the spikes 
class OffSpikes():
	def __call__(self, x):    
		x[x < 0] *= -1
		return x


class RateEncodeData():
	def __init__(self, nSteps, gain, offset):
		self.nSteps = nSteps
		self.gain = gain
		self.offset = offset
	
	def __call__(self, x):
		x = spikegen.rate(x, num_steps=nSteps, gain=self.gain, offset=self.offset)
		return x


class LatencyEncodeData():
	def __init__(self, nSteps, tau, threshold):
		self.nSteps = nSteps
		self.tau = tau
		self.threshold = threshold
	
	def __call__(self, x):
		x = spikegen.latency(x, num_steps=self.nSteps, 
												 tau=self.tau, threshold=self.threshold, 
												 linear=True, normalize=True, clip=True)
		return x


# repeat labels for nTimesteps
class CopyEncodeLabels():
	def __init__(self, nSteps):
		self.nSteps = nSteps
	
	def __call__(self, x):
		x = np.tile(x[None, :], (self.nSteps, 1))
		return x

copyRed = CopyRedChannel()

# SNN data
offSpikes = OffSpikes()

rate    = RateEncodeData(nSteps, 1, 0)
latency = LatencyEncodeData(nSteps, 5, 0.01)

# SNN labels
spikeLabels = CopyEncodeLabels(nSteps)

offRate    = transforms.Compose([offSpikes, rate])
offLatency = transforms.Compose([offSpikes, latency])



# *******************************************
# Training Helpers

def train_model(model, optimizer, lrScheduler, encoding, num_epochs, atol=1):
		# for each epoch... 
		# liveloss = PlotLosses()
		writer = SummaryWriter()

		best_model = None
		highest_val = 0

		for epoch in range(num_epochs):
			print('Training Model: ', model.__class__.__name__)
			print('Epoch {}/{}'.format(epoch, num_epochs - 1))
			print('-' * 10)
			# logs = {}

			# let every epoch go through one training cycle and one validation cycle
			for phase in ['train', 'val']:
				train_loss = 0
				correct = 0
				total = 0
				batch_idx = 0

				start_time = time.time()

				# SELECT PROPER MODE - train or val
				if phase == 'train':
					for param_group in optimizer.param_groups:
						print("LR", param_group['lr'])
					model.train()  # Set model to training mode
				else:
					model.eval()   # Set model to evaluate mode
				
				for inputs, labels in dataloaders[phase]:

					if encoding == True:
						labels = labels.permute((1, 0, 2))

					inputs = inputs.float()
					labels = labels.float()

					inputs = inputs.to(device)
					labels = labels.to(device)

					batch_idx += 1
					
					optimizer.zero_grad()
					
					with torch.set_grad_enabled(phase == 'train'):
						#    the above line says to disable gradient tracking for validation
						#    which makes sense since the model is in evaluation mode and we 
						#    don't want to track gradients for validation)  

						if encoding == True:
							mem, spikes, outputs = model(inputs)
						else:
							outputs = model(inputs)

						# compute loss where the loss function will be defined later
						if encoding == True:
							loss = loss_fn(outputs, labels[-1])
						else:
							loss = loss_fn(outputs, labels)

						# backward + optimize only if in training phase
						if phase == 'train':
							loss.backward()
							optimizer.step()

						train_loss += loss

						if encoding == True:
							# print(outputs.size(), labels.size())
							# print(outputs, labels)
							# c, t = numVectorsMatch(outputs[-1], labels[-1], 0, atol)
							c, t = numVectorsMatch(outputs, labels[-1], 0, atol)
						else:
							c, t = numVectorsMatch(outputs, labels, 0, atol)

						correct += c
						total   += t

						# print(correct, total)  

				# SCHEDULER COMMENTED OUT
				# lrScheduler.step(train_loss)

				# if phase == 'train':
				#   if  epoch%5 == 0:
				#   # prints for training and then validation (since the network will be in either train or eval mode at this point) 
				#     print(" Training Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

				# if phase == 'val' and epoch%5 == 0:
				#   print(" Validation Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

				prefix = ''
				if phase == 'val':
						prefix = 'val_'
						# tune.track.log(mean_accuracy=correct/total)

						if (correct/total) > highest_val:
							print('Best val till now: ', (correct/total))
							highest_val = (correct/total)
							# logs['what'] = highest_val
							best_model = copy.deepcopy(model)

				# logs[prefix + 'loss'] = train_loss.item()/(batch_idx)
				# logs[prefix + 'acc'] = (correct / total) * 100

				writer.add_scalar(f'loss/{phase}', train_loss.item()/(batch_idx), epoch)
				writer.add_scalar(f'accuracy/{phase}', (correct / total) * 100, epoch)

			# liveloss.update(logs)
			# liveloss.send()
			writer.flush()

		# end of single epoch iteration... repeat of n epochs  
		return best_model, model

def test_model(model, atol, encoding):

	total = correct = 0

	for inputs, labels in dataloaders['test']:

		if encoding == True:
			labels = labels.permute((1, 0, 2))

		inputs = inputs.float()
		labels = labels.float()

		inputs = inputs.to(device)
		labels = labels.to(device)

		if encoding == True:
			mem, spikes, outputs = model(inputs)
		else:
			outputs = model(inputs)

		# print("outputs")
		# print(outputs.cpu().detach().numpy()[:5])
		
		# print("labels")
		# print(labels.cpu().detach().numpy()[:5])

		# print("huh", outputs[-1], labels[-1])

		if encoding == True:
			c, t = numVectorsMatch(outputs, labels[-1], 0, atol)
		else:
			c, t = numVectorsMatch(outputs, labels, 0, atol)

		correct += c
		total   += t

	print((correct/total) * 100)

	if encoding == True:
		return mem, spikes, outputs
	else:
		return outputs

def numVectorsMatch(outputs, labels, rtol = 0, atol = 0.5):

	correct = 0
	total   = labels.size(0)
	nDim    = labels.size(1)

	close = torch.isclose(outputs, labels, rtol, atol)

	for r in close:
		if torch.sum(r == True) == nDim:
			correct += 1

	return correct, total

loss_fn = nn.MSELoss()

def pipeline(model, epochs=25, lr=1e-2, weight_decay=0.1, encoding=False, patience=5, atol=1):

	model.to(torch.float)
	model.to(device)

	opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # TODO: tune betas
	lrSched = ReduceLROnPlateau(opt, patience=patience)
	
	best_model, m = train_model(model, opt, lrSched, encoding, num_epochs=epochs, atol=atol)

	return best_model, m

def getAtol(m, encoding=False):

	m.to(torch.float)
	m.to(device)
	m.eval()

	atols = [5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

	for atol in atols:
		print(atol)
		outputs = test_model(m, atol=atol, encoding=encoding)

# *******************************************
# Models

class LCN(nn.Module):
		def __init__(self, in_dim, out_dim, K, factor, num_layer, use_cuda=True):
				super(LCN, self).__init__()
				# Initialize parameters
				self.dtype = torch.FloatTensor
				self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
				self.knn_list = []
				self.num_layer = num_layer
				self.use_cuda = use_cuda

				# Initialize weight, bias and KNN data
				dim = in_dim
				for i in range(num_layer):
						# dim = int(round(dim / factor))
						dim = int(math.floor(dim / factor))

						# Weight and bias
						w = torch.Tensor(dim, K).zero_().type(self.dtype).normal_(0, (2.0 / K) ** 0.5)
						b = torch.zeros(1, dim).type(self.dtype)
						self.weight_param.append(torch.nn.Parameter(w, requires_grad=True))
						self.bias_param.append(torch.nn.Parameter(b, requires_grad=True))

						# KNN
						h5f = h5py.File('KNN/%d/knn_index_%d.h5' % (K, i), 'r')
						k_nearest = torch.from_numpy(h5f['data'][:]).type(torch.long)
						h5f.close()

						self.knn_list.append(k_nearest)

				self.fc_angle = nn.Linear(dim, out_dim)
				# self.fc_class = nn.Linear(dim, 50)

		def forward(self, input):
				# Input size (Batch size, num_features)
				x = input

				# print("0", x.shape)

				batch_size = input.shape[0]
				for i in range(self.num_layer):
						# print(len(self.weight_param), len(self.bias_param), len(self.knn_list))
						if self.use_cuda:
							weight, bias, knn = self.weight_param[i], self.bias_param[i], self.knn_list[i].cuda()
						else:
							weight, bias, knn = self.weight_param[i], self.bias_param[i], self.knn_list[i]
						# print("1", x.unsqueeze(1).shape)
						x = x.unsqueeze(1).expand(-1, weight.shape[0], -1)
						# print("1", x.shape, knn.shape)
						knn = knn.unsqueeze(0).expand(batch_size, -1, -1)
						# print("2", knn.shape)
						# print(x.shape, knn.shape)
						x = torch.gather(x, 2, knn)
						# print("3", x.shape, weight.unsqueeze(0).expand(batch_size, -1, -1).shape)
						# print(x.get_device())
						# print(weight.get_device())
						x = x * weight.unsqueeze(0).expand(batch_size, -1, -1)
						x = F.relu(torch.sum(x, 2) + bias)
						del weight, bias, knn

				angle = self.fc_angle(x)
				# classification = self.fc_class(x)
				return angle

class LCNSpiking(nn.Module):
		def __init__(self, in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda=True, spikeGrad=None, inhibition=False):
				super(LCNSpiking, self).__init__()

				# Initialize parameters
				self.dtype = torch.FloatTensor
				self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
				self.knn_list  = []
				self.num_layer = num_layer
				self.use_cuda  = use_cuda

				self.alpha = alpha
				self.beta  = beta

				# Initialize weight, bias, spiking neurons, and KNN data
				dim = in_dim
				for i in range(num_layer):
						dim = int(math.floor(dim / factor))

						# Weight and bias
						w = torch.Tensor(dim, K).zero_().type(self.dtype).normal_(0, (2.0 / K) ** 0.5)
						b = torch.zeros(1, dim).type(self.dtype)
						self.weight_param.append(torch.nn.Parameter(w, requires_grad=True))
						self.bias_param.append(torch.nn.Parameter(b, requires_grad=True))

						# KNN
						h5f = h5py.File('KNN/%d/knn_index_%d.h5' % (K, i), 'r')
						k_nearest = torch.from_numpy(h5f['data'][:]).type(torch.long)
						h5f.close()

						self.knn_list.append(k_nearest)

						# Spiking Neurons
						# lif = snn.Synaptic(alpha=alpha, beta=beta)
						# self.spike_param[f'{i}'] = lif

				# Spiking Neurons
				self.lif0 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition) # init_hidden=True, output=True)
				self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition)
				self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition)
				self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition)
				self.lif4 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition)
			
				# Make sure dictionary doesn't stop learning
				self.spike_param = {
						0: self.lif0, 
						1: self.lif1, 
						2: self.lif2, 
						3: self.lif3, 
						4: self.lif4, 
				}     

				# Output
				self.fc_angle = nn.Linear(dim, out_dim)


		def forward(self, input):

				# synapses  = []
				# membranes = []

				# for i in range(self.num_layer):
				#   syn, mem = self.spike_param[f'{i}'].init_synaptic()
				#   synapses.append(syn)
				#   membranes.append(mem)

				# Initialize hidden states and outputs at t=0
				syn0, mem0 = self.lif0.init_synaptic()
				syn1, mem1 = self.lif1.init_synaptic()
				syn2, mem2 = self.lif2.init_synaptic()
				syn3, mem3 = self.lif3.init_synaptic()
				syn4, mem4 = self.lif4.init_synaptic()

				synapses = {
						0: syn0, 
						1: syn1, 
						2: syn2, 
						3: syn3, 
						4: syn4, 
				}

				membranes = {
						0: mem0, 
						1: mem1, 
						2: mem2, 
						3: mem3, 
						4: mem4, 
				}

				# TODO: Record the final layer?
				mem4_rec = []
				spk4_rec = []

				batch_size = input.shape[0]

				input = input.permute(1, 0, 2)  # (nSteps, batch, data)
				x     = None
				angle = None

				for step in range(nSteps):
					x = input[step]    

					for i in range(self.num_layer):
							# print(len(self.weight_param), len(self.bias_param), len(self.knn_list))
							if self.use_cuda:
								weight, bias, knn = self.weight_param[i], self.bias_param[i], self.knn_list[i].cuda()
							else:
								weight, bias, knn = self.weight_param[i], self.bias_param[i], self.knn_list[i]
							# print("1", x.unsqueeze(1).shape)
							x = x.unsqueeze(1).expand(-1, weight.shape[0], -1)
							# print("1", x.shape, knn.shape)
							knn = knn.unsqueeze(0).expand(batch_size, -1, -1)
							# print("2", knn.shape)
							# print(x.shape, knn.shape)
							x = torch.gather(x, 2, knn)
							# print("3", x.shape, weight.unsqueeze(0).expand(batch_size, -1, -1).shape)
							x = x * weight.unsqueeze(0).expand(batch_size, -1, -1)

							# print("1", x.shape)
							# x = F.relu(torch.sum(x, 2) + bias)
							_, _, x = self.spike_param[i](torch.sum(x, 2) + bias, synapses[i], membranes[i])
							# print("2", x.shape)
							del weight, bias, knn

					angle = self.fc_angle(x)
					# print("out", angle.shape)

				return mem4_rec, spk4_rec, angle

def main():

	global device, dataloaders

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

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

	# 100 epochs, normal vs delta data on normal LCN
	"""
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

	getAtol(model)
	# atol  1e-5 1e-4 1e-3  1e-2  1e-1  0.5
	#       0    0    0.44  56.7  100   100
	"""

	# model = LCN(14400, 2, 15, 2, 5, True)
	# model.load_state_dict(torch.load('linet_deltaData_100epoch'))


	m = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	best, last = pipeline(m, epochs=100, lr=1e-3, weight_decay=0, encoding=True, patience=7, atol=1e-2)
	torch.save(last.state_dict(), 'linet_spiking_rate_20steps')


if __name__ == "__main__":
	main()