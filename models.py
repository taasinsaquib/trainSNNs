import math
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn

from data import nSteps

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
		def __init__(self, in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda=True, spikeGrad=None, inhibition=False, directOutput=False):
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
				self.lif0 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract') # init_hidden=True, output=True)
				self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract')
				self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract')
				self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract')
				self.lif4 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spikeGrad, inhibition=inhibition, reset_mechanism='subtract')
			
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

					if not directOutput:
						angle = self.fc_angle(x)
					else:
						angle = x
					# print("out", angle.shape)

				return mem4_rec, spk4_rec, angle

class LCNSpikingHybrid(nn.Module):
  def __init__(self, num_spiking, in_dim, out_dim, K, factor, num_layer, alpha, beta, use_cuda=True, spikeGrad=None, inhibition=False):
      super(LCNSpikingHybrid, self).__init__()

      # SNN PART
      self.num_spiking = num_spiking
      self.snn = LCNSpiking(in_dim, out_dim, K, factor, num_spiking, alpha, beta, use_cuda, spikeGrad, inhibition)
      
      # ANN PART
      self.dtype = torch.FloatTensor
      self.weight_param, self.bias_param = nn.ParameterList(), nn.ParameterList()
      self.knn_list = []
      self.num_layer = num_layer
      self.use_cuda = use_cuda
    
      dim = in_dim / (factor ** nSpiking)

      # Initialize weight, bias, spiking neurons, and KNN data
      for i in range(num_spiking, num_layer):
          dim = int(math.floor(dim / factor))

          # Weight and bias
          w = torch.Tensor(dim, K).zero_().type(self.dtype).normal_(0, (2.0 / K) ** 0.5)
          b = torch.zeros(1, dim).type(self.dtype)
          self.weight_param.append(torch.nn.Parameter(w, requires_grad=True))
          self.bias_param.append(torch.nn.Parameter(b, requires_grad=True))

          # KNN
          h5f = h5py.File('./KNN/%d/knn_index_%d.h5' % (K, i), 'r')
          k_nearest = torch.from_numpy(h5f['data'][:]).type(torch.long)
          h5f.close()

          self.knn_list.append(k_nearest)

      self.fc_angle = nn.Linear(dim, out_dim)

  def forward(self, input):

    x = input
    batch_size = input.shape[0]

    # SNN PART
    _, _, x = self.snn(x)

    print(x.size())

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
    print(angle.size())
    # classification = self.fc_class(x)
    return angle


# TODO
"""
	larger models with more layers
	
	make models where there are varying numbers of spiking layers
		change front and back, spiking and non-spiking

	model that encodes 2 channels

	loss function looks at all timesteps? 
"""

def main():

	linet          = LCN(14400, 2, 15, 2, 5, True)
	spiking_linet  = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	spiking_hybrid = LCNSpiking4L(4, 14400, 2, 15, 2, 5, 0, 0.25, True)

if __name__ == "__main__":
	main()