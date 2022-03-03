import math
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn

from data import nSteps, batchSize


# *****************************************************************************
# ANNs
# *****************************************************************************

class FC(nn.Module):
	def __init__(self):
		super(FC, self).__init__()

		self.fc0 = nn.Linear(14400, 7200)
		self.fc1 = nn.Linear(7200, 3600)
		self.fc2 = nn.Linear(3600, 1800)
		self.fc3 = nn.Linear(1800, 900)
		self.fc4 = nn.Linear(900, 450)
		self.fc5 = nn.Linear(450, 2)

	def forward(self, x):

		x = self.fc0(x)
		x = torch.relu(x)

		x = self.fc1(x)
		x = torch.relu(x)

		x = self.fc2(x)
		x = torch.relu(x)

		x = self.fc3(x)
		x = torch.relu(x)

		x = self.fc4(x)
		x = torch.relu(x)

		x = self.fc5(x)
		
		return x


# same as above, no biases to train (for converting to SNN)
# didn't get good enough results to convert it
class FCnoBias(nn.Module):
	def __init__(self):
		super(FCnoBias, self).__init__()

		self.fc0 = nn.Linear(14400, 7200, bias=False)
		self.fc1 = nn.Linear(7200, 3600, bias=False)
		self.fc2 = nn.Linear(3600, 1800, bias=False)
		self.fc3 = nn.Linear(1800, 900, bias=False)
		self.fc4 = nn.Linear(900, 450, bias=False)
		self.fc5 = nn.Linear(450, 2, bias=False)

	def forward(self, x):

		x = self.fc0(x)
		x = torch.relu(x)

		x = self.fc1(x)
		x = torch.relu(x)

		x = self.fc2(x)
		x = torch.relu(x)

		x = self.fc3(x)
		x = torch.relu(x)

		x = self.fc4(x)
		x = torch.relu(x)

		x = self.fc5(x)
		
		return x


class LCN(nn.Module):
	def __init__(self, in_dim, out_dim, K, factor, num_layer, use_cuda=True, directOutput=False):
		super(LCN, self).__init__()
		# Initialize parameters
		self.dtype = torch.FloatTensor
		self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
		self.knn_list = []
		self.num_layer = num_layer
		self.use_cuda = use_cuda
		self.directOutput = directOutput

		# Initialize weight, bias and KNN data
		dim = in_dim
		for i in range(num_layer):
			dim = int(math.floor(dim / factor))

			# Weight and bias
			w = torch.Tensor(dim, K).zero_().type(self.dtype).normal_(0, (2.0 / K) ** 0.5)
			b = torch.zeros(1, dim).type(self.dtype)
			self.weight_param.append(torch.nn.Parameter(w, requires_grad=True))
			self.bias_param.append(torch.nn.Parameter(b, requires_grad=True))

			# KNN
			h5f = h5py.File('KNN/%d/%d/%d/knn_index_%d.h5' % (in_dim, factor, K, i), 'r')
			k_nearest = torch.from_numpy(h5f['data'][:]).type(torch.long)
			h5f.close()

			self.knn_list.append(k_nearest)

		self.fc_angle = nn.Linear(dim, out_dim)

	def forward(self, input):
		# Input size (Batch size, num_features)
		x = input

		# print("0", x.size())
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

		if self.directOutput:
			angle = x
		else:
			angle = self.fc_angle(x)

		return angle


# same as above, no biases to train (for converting to SNN)
# didn't get good enough results to convert it
class LCNnoBias(nn.Module):
	def __init__(self, in_dim, out_dim, K, factor, num_layer, use_cuda=True, directOutput=False):
		super(LCNnoBias, self).__init__()
		# Initialize parameters
		self.dtype = torch.FloatTensor
		self.weight_param= nn.ParameterList()
		self.knn_list = []
		self.num_layer = num_layer
		self.use_cuda = use_cuda
		self.directOutput = directOutput

		# Initialize weight, bias and KNN data
		dim = in_dim
		for i in range(num_layer):
			dim = int(math.floor(dim / factor))

			# Weight and bias
			w = torch.Tensor(dim, K).zero_().type(self.dtype).normal_(0, (2.0 / K) ** 0.5)
			self.weight_param.append(torch.nn.Parameter(w, requires_grad=True))

			# KNN
			h5f = h5py.File('KNN/%d/knn_index_%d.h5' % (K, i), 'r')
			k_nearest = torch.from_numpy(h5f['data'][:]).type(torch.long)
			h5f.close()

			self.knn_list.append(k_nearest)

		self.fc_angle = nn.Linear(dim, out_dim)

	def forward(self, input):
		# Input size (Batch size, num_features)
		x = input

		# print("0", x.shape)
		batch_size = input.shape[0]
		for i in range(self.num_layer):
			# print(len(self.weight_param), len(self.bias_param), len(self.knn_list))
			if self.use_cuda:
				weight, knn = self.weight_param[i], self.knn_list[i].cuda()
			else:
				weight, knn = self.weight_param[i], self.knn_list[i]
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
			x = F.relu(torch.sum(x, 2))
			del weight, knn

		if self.directOutput:
			angle = x
		else:
			angle = self.fc_angle(x)

		return angle


# for lens and pupil controllers
# haven't gotten the data together to train it yet
class ShallowNN(nn.Module):
	def __init__(self, in_dim):
		super(ShallowNN, self).__init__()

		self.fc1 = nn.Linear(in_dim, 2000)
		self.fc2 = nn.Linear(2000, 1000)
		self.fc3 = nn.Linear(1000, 1)

	def forward(self, input):
		x = input

		x = self.fc1(x)
		x = torch.relu(x)

		x = self.fc2(x)
		x = torch.relu(x)

		x = self.fc3(x)

		return x


# *****************************************************************************
# SNNs
# *****************************************************************************

class FCSpiking(nn.Module):
	def __init__(self, alpha, beta, spikeGrad=None):
		super(FCSpiking, self).__init__()

		self.fc0 = nn.Linear(14400, 3600)
		self.fc1 = nn.Linear(3600, 900)
		self.fc2 = nn.Linear(900, 450)
		self.fc3 = nn.Linear(450, 2)

		self.lif0 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad)   #, init_hidden=True, output=True)
		self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad)   #, init_hidden=True, output=True)
		self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad)   #, init_hidden=True, output=True) 

	def forward(self, input):
		
		input = input.permute(1, 0, 2)  # (nSteps, batch, data)

		for step in range(nSteps):
			x = input[step]    

			syn0, mem0 = self.lif0.init_synaptic()
			syn1, mem1 = self.lif1.init_synaptic()
			syn2, mem2 = self.lif2.init_synaptic()

			x = self.fc0(x)
			_, _, x = self.lif0(x, syn0, mem0)

			x = self.fc1(x)
			_, _, x = self.lif0(x, syn1, mem1)

			x = self.fc2(x)
			_, _, x = self.lif0(x, syn2, mem2)

			x = self.fc3(x)
		
		return x


# keeping this here for legacy purposes
# its basically an RNN since we're passing the threshold voltage at every time step
# LCNSpiking2 fixes this
class LCNSpiking(nn.Module):
		def __init__(self, in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda=True, spikeGrad=None, inhibition=False, directOutput=False):
				super(LCNSpiking, self).__init__()

				# Initialize parameters
				self.dtype = torch.FloatTensor
				self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
				self.knn_list  = []
				self.num_layer = num_layer
				self.use_cuda  = use_cuda
				self.directOutput = directOutput

				self.alpha = alpha
				self.beta  = beta

				self.thresholds = []

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
						self.thresholds.append(torch.rand(dim))

				# temporary, to allow for hybrid model training to be automated
				for i in range(num_layer, 5):
					self.thresholds.append(torch.rand(dim))


				# Spiking Neurons
				self.lif0 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[0], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[1], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[2], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[3], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif4 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[4], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
			
				self.spike_param = {
						0: self.lif0, 
						1: self.lif1, 
						2: self.lif2, 
						3: self.lif3, 
						4: self.lif4,
				}     

				# Output
				self.fc_angle = nn.Linear(dim, out_dim)

				self.nStepBackprop = 20

		def forward(self, input):
				
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

				input  = input.permute(1, 0, 2)  # (nSteps, batch, data)
				x      = None
				# angles2 = torch.zeros((self.nStepBackprop, batch_size, 2))	# add if-else statement for cpu training)
				angles = None

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

					if self.directOutput:
						angle = x
					else:
						angle = self.fc_angle(x)
					

					"""
					j = nSteps - self.nStepBackprop 
					if j >= 0:
						angles[j] = angle
					"""
					angles = angle
					# angles2[step] = angle

				return mem4_rec, spk4_rec, angles


# fixes LCNSpiking to actually be an SNN, just by modifying the use of spike_param[i] in forward
class LCNSpiking2(nn.Module):
		def __init__(self, in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda=True, spikeGrad=None, inhibition=False, directOutput=False):
				super(LCNSpiking2, self).__init__()

				# Initialize parameters
				self.dtype = torch.FloatTensor
				self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
				self.knn_list  = []
				self.num_layer = num_layer
				self.use_cuda  = use_cuda
				self.directOutput = directOutput

				self.alpha = alpha
				self.beta  = beta

				self.thresholds = []

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
						h5f = h5py.File('KNN/%d/%d/%d/knn_index_%d.h5' % (in_dim, factor, K, i), 'r')
						k_nearest = torch.from_numpy(h5f['data'][:]).type(torch.long)
						h5f.close()

						self.knn_list.append(k_nearest)

						# Spiking Neurons
						self.thresholds.append(torch.rand(dim))

				# temporary, to allow for hybrid model training to be automated
				for i in range(num_layer, 5):
					self.thresholds.append(torch.rand(dim))


				# Spiking Neurons
				self.lif0 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[0], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[1], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[2], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[3], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif4 = snn.Synaptic(alpha=alpha, beta=beta, threshold=self.thresholds[4], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
			
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

				self.nStepBackprop = 20

		def forward(self, input):
				
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

				input  = input.permute(1, 0, 2)  # (nSteps, batch, data)
				x      = None
				# angles2 = torch.zeros((nSteps, batch_size, 2)).cuda()	# add if-else statement for cpu training)
				# angles = []

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

							if i == self.num_layer-1:
								_, _, membranes[i] = self.spike_param[i](torch.sum(x, 2) + bias, synapses[i], membranes[i])
								x = membranes[i]
							else:
								x, _, membranes[i] = self.spike_param[i](torch.sum(x, 2) + bias, synapses[i], membranes[i])

							# print("2", x.shape)
							del weight, bias, knn

					if self.directOutput:
						angle = x
					else:
						angle = self.fc_angle(x)
						# angles2[step] = angle

					"""
					j = nSteps - self.nStepBackprop 
					if j >= 0:
						angles[j] = angle
					"""
					angles = angle
					# angles2[step] = angle

				return mem4_rec, spk4_rec, angles

# Same as LCNSpiking2, but using Leaky neurons instead of synaptic
class LCNSpiking2_Leaky(nn.Module):
		def __init__(self, in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda=True, spikeGrad=None, inhibition=False, directOutput=False):
				super(LCNSpiking2, self).__init__()

				# Initialize parameters
				self.dtype = torch.FloatTensor
				self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
				self.knn_list  = []
				self.num_layer = num_layer
				self.use_cuda  = use_cuda
				self.directOutput = directOutput

				self.alpha = alpha
				self.beta  = beta

				self.thresholds = []

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
						self.thresholds.append(torch.rand(dim))

				# temporary, to allow for hybrid model training to be automated
				for i in range(num_layer, 5):
					self.thresholds.append(torch.rand(dim))


				# Spiking Neurons
				self.lif0 = snn.Leaky(beta=beta, threshold=self.thresholds[0], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif1 = snn.Leaky(beta=beta, threshold=self.thresholds[1], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif2 = snn.Leaky(beta=beta, threshold=self.thresholds[2], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif3 = snn.Leaky(beta=beta, threshold=self.thresholds[3], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
				self.lif4 = snn.Leaky(beta=beta, threshold=self.thresholds[4], spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract', learn_threshold=True)
			
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

				self.nStepBackprop = 20

		def forward(self, input):
				
				# Initialize hidden states and outputs at t=0
				mem0 = self.lif0.init_leaky()
				mem1 = self.lif1.init_leaky()
				mem2 = self.lif2.init_leaky()
				mem3 = self.lif3.init_leaky()
				mem4 = self.lif4.init_leaky()

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

				input  = input.permute(1, 0, 2)  # (nSteps, batch, data)
				x      = None
				# angles2 = torch.zeros((nSteps, batch_size, 2)).cuda()	# add if-else statement for cpu training)
				angles = None

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
							if i == self.num_layer-1:
								_, membranes[i] = self.spike_param[i](torch.sum(x, 2) + bias, membranes[i])
								x = membranes[i]
							else:
								x, membranes[i] = self.spike_param[i](torch.sum(x, 2) + bias, membranes[i])

							# print("2", x.shape)
							del weight, bias, knn

					if self.directOutput:
						angle = x
					else:
						angle = self.fc_angle(x)
					

					"""
					j = nSteps - self.nStepBackprop 
					if j >= 0:
						angles[j] = angle
					"""
					angles = angle
					# angles2[step] = angle

				return mem4_rec, spk4_rec, angles


# *****************************************************************************
# Hybrid SNN-ANN
# *****************************************************************************

# first layers are SNN, last layers are ANN
# still uses LCN neighbor matrices
class LCNSpikingHybrid(nn.Module):
	def __init__(self, num_spiking, in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda=True, spikeGrad=None, inhibition=False):
			super(LCNSpikingHybrid, self).__init__()

			# SNN PART
			self.num_spiking = num_spiking
			self.snn = LCNSpiking2(in_dim, out_dim, K, factor, num_spiking, alpha, beta, use_cuda, spikeGrad, inhibition, directOutput=True)
			
			# ANN PART
			self.dtype = torch.FloatTensor
			self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
			self.knn_list = []
			self.num_layer = num_layer
			self.use_cuda = use_cuda
		
			dim = in_dim / (factor ** self.num_spiking)

			# Initialize weight, bias, spiking neurons, and KNN data
			for i in range(num_spiking, num_layer):
					dim = int(math.floor(dim / factor))

					# Weight and bias
					w = torch.Tensor(dim, K).zero_().type(self.dtype).normal_(0, (2.0 / K) ** 0.5)
					b = torch.zeros(1, dim).type(self.dtype)
					self.weight_param.append(torch.nn.Parameter(w, requires_grad=True))
					self.bias_param.append(torch.nn.Parameter(b, requires_grad=True))

					# KNN
					h5f = h5py.File('KNN/%d/%d/%d/knn_index_%d.h5' % (in_dim, factor, K, i), 'r')
					k_nearest = torch.from_numpy(h5f['data'][:]).type(torch.long)
					h5f.close()

					self.knn_list.append(k_nearest)

			self.fc_angle = nn.Linear(dim, out_dim)

	def forward(self, input):

		x = input
		batch_size = input.shape[0]

		# SNN PART
		_, _, x = self.snn(x)

		# ANN PART
		for i in range(0, self.num_layer-self.num_spiking):
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

# only have a linear layer at the end of the SNN
class LCNSpikingHybrid2(nn.Module):
	def __init__(self, num_spiking, in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda=True, spikeGrad=None, inhibition=False):
			super(LCNSpikingHybrid2, self).__init__()

			# SNN PART
			self.num_spiking = num_spiking
			self.snn = LCNSpiking2(in_dim, out_dim, K, factor, num_spiking, alpha, beta, use_cuda, spikeGrad, inhibition, directOutput=True)
			
			# ANN PART
			self.dtype = torch.FloatTensor
			self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
			self.knn_list = []
			self.num_layer = num_layer
			self.use_cuda = use_cuda
		
			dim = int(in_dim / (factor ** self.num_spiking))
			self.fc_angle = nn.Linear(dim, out_dim)

	def forward(self, input):

		x = input

		# SNN PART
		_, _, x = self.snn(x)

		angle = self.fc_angle(x)
		
		return angle


class LCNSpikingHybrid3(nn.Module):
	def __init__(self, num_spiking, in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda=True, spikeGrad=None, inhibition=False):
			super(LCNSpikingHybrid3, self).__init__()

			# SNN PART
			self.num_spiking = num_spiking
			self.snn = LCNSpiking2(in_dim, out_dim, K, factor, num_spiking, alpha, beta, use_cuda, spikeGrad, inhibition, directOutput=True)
			
			# ANN PART
			self.dtype = torch.FloatTensor
			self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
			self.knn_list = []
			self.num_layer = num_layer
			self.use_cuda = use_cuda
		
			dim = in_dim / (factor ** self.num_spiking)

			# Initialize weight, bias, spiking neurons, and KNN data
			for i in range(num_spiking, num_layer):
					dim = int(math.floor(dim / factor))

					# Weight and bias
					w = torch.Tensor(dim, K).zero_().type(self.dtype).normal_(0, (2.0 / K) ** 0.5)
					b = torch.zeros(1, dim).type(self.dtype)
					self.weight_param.append(torch.nn.Parameter(w, requires_grad=True))
					self.bias_param.append(torch.nn.Parameter(b, requires_grad=True))

					# KNN
					h5f = h5py.File('KNN/%d/%d/%d/knn_index_%d.h5' % (in_dim, factor, K, i), 'r')
					k_nearest = torch.from_numpy(h5f['data'][:]).type(torch.long)
					h5f.close()

					self.knn_list.append(k_nearest)

			self.fc_angle = nn.Linear(dim, out_dim)

	def forward(self, input):

		x = input
		batch_size = input.shape[0]

		# SNN PART
		_, _, x = self.snn(x)

		# ANN PART
		for i in range(0, self.num_layer-self.num_spiking):
				# print(len(self.weight_param), len(self.bias_param), len(self.knn_list))
				if self.use_cuda:
						weight, bias, knn = self.weight_param[i], self.bias_param[i], self.knn_list[i].cuda()
				else:
						weight, bias, knn = self.weight_param[i], self.bias_param[i], self.knn_list[i]

				x = x.unsqueeze(1).expand(-1, weight.shape[0], -1)

				knn = knn.unsqueeze(0).expand(batch_size, -1, -1)

				x = torch.gather(x, 2, knn)

				x = x * weight.unsqueeze(0).expand(batch_size, -1, -1)
				x = torch.sum(x, 2) + bias

				del weight, bias, knn

		angle = self.fc_angle(x)

		return angle

# *****************************************************************************
# Other
# *****************************************************************************

# One LCN for positive delta, one for negative
# TODO: turn LCNSPiking into LCNSpiking2
class LCNChannelStack(nn.Module):
	def __init__(self, in_dim, out_dim, K, factor, num_layer, use_cuda=True, spiking=False, alpha=0, beta=1):
		super(LCNChannelStack, self).__init__()

		self.in_dim = in_dim
		self.spiking = spiking

		if self.spiking is False:
			self.nnPos = LCN(in_dim, out_dim, K, factor, num_layer, use_cuda, directOutput=False)
			self.nnNeg = LCN(in_dim, out_dim, K, factor, num_layer, use_cuda, directOutput=False)
		else:
			self.nnPos = LCNSpiking(in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda, directOutput=False)
			self.nnNeg = LCNSpiking(in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda, directOutput=False)
		
		self.fc3 = nn.Linear(4, out_dim)

	def forward(self, input):
		x = input
		batch_size = x.shape[0]

		if self.spiking is False:
			xPos = self.nnPos(x[:, :self.in_dim])
			xNeg = self.nnNeg(x[:, self.in_dim:])
		else:
			_, _, xPos = self.nnPos(x[:, :, :self.in_dim])
			_, _, xNeg = self.nnNeg(x[:, :, self.in_dim:])	

		x = torch.hstack((xPos, xNeg))
		x = x.view((batch_size, -1))

		"""
		# print("x1", x.size())

		x = self.fc1(x)
		x = torch.relu(x)

		# print("x2", x.size())

		x = self.fc2(x)
		"""

		x = torch.relu(x)
		x = self.fc3(x)

		return x


# *****************************************************************************
# For snn_toolbox Conversion
# *****************************************************************************

# FC
# """
class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.input_shape = (1, 14400)

		self.fc0 = nn.Linear(14400, 450)
		# self.fc1 = nn.Linear(7200, 3600)
		# self.fc2 = nn.Linear(3600, 1800)
		# self.fc3 = nn.Linear(1800, 900)
		# self.fc4 = nn.Linear(900, 450)
		self.fc5 = nn.Linear(450, 2)

	def forward(self, x):

		x = self.fc0(x)
		x = torch.relu(x)

		# x = self.fc1(x)
		# x = torch.relu(x)

		# x = self.fc2(x)
		# x = torch.relu(x)

		# x = self.fc3(x)
		# x = torch.relu(x)

		# x = self.fc4(x)
		# x = torch.relu(x)

		x = self.fc5(x)
		
		return x
# """

# LCN
"""
class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		in_dim = 14400
		out_dim = 2
		K = 25
		factor = 2
		num_layer = 5

		self.input_shape = (1, 14400)

		# Initialize parameters
		self.dtype = torch.FloatTensor
		self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
		self.knn_list = []
		self.num_layer = 5
		self.use_cuda = False
		self.directOutput = False

		# Initialize weight, bias and KNN data
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

		self.fc_angle = nn.Linear(dim, out_dim)

	def forward(self, input):
		# Input size (Batch size, num_features)
		x = input

		# assuming first dim is time
		# for i in range(0, input.size()[0]):

			# x = input[i]
		# print(input.size(), x.size())
		x = input[0]		 # confused about shape of input

		# print("0", x.shape)
		batch_size = input.shape[0]
		for i in range(self.num_layer):
			# print(len(self.weight_param), len(self.bias_param), len(self.knn_list))
			if self.use_cuda:
				weight, bias, knn = self.weight_param[i], self.bias_param[i], self.knn_list[i].cuda()
			else:
				weight, bias, knn = self.weight_param[i], self.bias_param[i], self.knn_list[i]
			# print("1", x.unsqueeze(1).shape)
			x = x.unsqueeze(1).expand(-1, -1, weight.shape[0], -1)
			# print("1", x.shape, knn.shape)
			knn = knn.unsqueeze(0).expand(batch_size, -1, -1)
			# print("2", knn.shape)
			# print(x.size(), knn.size())
			x = torch.gather(x, 2, knn)
			# print("3", x.shape, weight.unsqueeze(0).expand(batch_size, -1, -1).shape)
			# print(x.get_device())
			# print(weight.get_device())
			x = x * weight.unsqueeze(0).expand(batch_size, -1, -1)
			x = F.relu(torch.sum(x, 2) + bias)
			del weight, bias, knn

		if self.directOutput:
			angle = x
		else:
			angle = self.fc_angle(x)

		return angle
"""

"""
class Model(nn.Module):
		def __init__(self):
			super(Model, self).__init__()

			# The input_shape field is required by SNN toolbox.
			self.input_shape = (1, 28, 28)

			layers_trunk = [
					nn.Conv2d(1, 16, kernel_size=5, stride=2),
					# BatchNorm doesn't work with Keras==2.3.1 because for some reason
					# they put the batch-norm axis in a list.
					# nn.BatchNorm2d(16),
					nn.ReLU(),
					nn.AvgPool2d(kernel_size=2, stride=2)]
			layers_branch1 = [
					nn.Conv2d(16, 32, kernel_size=3, padding=1),
					nn.ReLU()]
			layers_branch2 = [
					nn.Conv2d(16, 8, kernel_size=1),
					nn.ReLU()]
			layers_head = [
					nn.Conv2d(40, 8, kernel_size=1),
					nn.ReLU()]
			layers_classifier = [
					nn.Dropout(1e-5),
					nn.Linear(288, 10),
					nn.Softmax(1)]
			self.trunk = nn.Sequential(*layers_trunk)
			self.branch1 = nn.Sequential(*layers_branch1)
			self.branch2 = nn.Sequential(*layers_branch2)
			self.head = nn.Sequential(*layers_head)
			self.classifier = nn.Sequential(*layers_classifier)

		def forward(self, x):
			print(x.size())
			x = self.trunk(x)
			x1 = self.branch1(x)
			x2 = self.branch2(x)
			x = torch.cat([x1, x2], 1)
			x = self.head(x)
			x = x.view(-1, 288)  # Flatten
			x = self.classifier(x)
			return x
"""

# *****************************************************************************
# main
# *****************************************************************************

def main():

	linet          = LCN(14400, 2, 15, 2, 5, True)
	spiking_linet  = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	spiking_hybrid = LCNSpiking4L(4, 14400, 2, 15, 2, 5, 0, 0.25, True)

	pupilNN = ShallowNN(14400)

	#  test model
	m = LCNSpiking(14400, 2, 15, 2, 5, 0, 1, True)
	m.to(torch.float)
	m.to(device)

	d = data[0]
	l = labels[0]

	d = torch.from_numpy(d).float()
	l = torch.from_numpy(l).float()

	d = rate(d)
	d = d[None, :]
	l = l[None, :]

	print(d.shape, l.shape)
	print(device)

	d = d.to(device)
	l = l.to(device)

	_, _, out = m(d)


	loss_fn = nn.MSELoss()
	loss = 0
	for i in range(0, nSteps):
		loss += loss_fn(out[i], l)
	loss /= nSteps

	print(loss.cpu().detach().numpy())


if __name__ == "__main__":
	main()
