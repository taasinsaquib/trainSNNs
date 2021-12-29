import numpy  as np
import pandas as pd

import torch
from   torch.utils.data import Dataset, TensorDataset, random_split

from   torchvision import transforms

from   snntorch import spikegen

# *******************************************
# Constants

seed = 528
np.random.seed(seed)
torch.manual_seed(seed)

batchSize = 8
nSteps = 20


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


# *******************************************
# Data Functions

def loadData(dir, name='', eyePart='Foveation'):
	data   = np.load(f'./{dir}/data{name}.npy',   mmap_mode='r+')
	labels = np.load(f'./{dir}/labels{name}.npy', mmap_mode='r+')  # TODO: turn into degrees or keep radians?

	print(f'Data shape: {data.shape}, Labels shape: {labels.shape}')

	subtractMean(labels, eyePart)
	normalization(labels, eyePart)

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
# Mean and Std Dev of labels

def subtractMean(input, eyePart):
	meanValue = np.mean(input, axis=0)
	input -= np.mean(input,axis = 0)

	print("Mean", meanValue)
	
	np.savetxt("./stats/mean" + eyePart + ".csv", meanValue, delimiter=",")

	return input

# Nomralization by dividing the standard variation
def normalization(input, eyePart):
	stdValue =np.std(input, axis = 0)
	input /= stdValue

	print("STD", stdValue)

	np.savetxt("./stats/std" + eyePart + ".csv", stdValue, delimiter=",")
	
	return input



def main():

	"""
	# sigDir = "training_data/siggraph_data"
	data, labels = loadData(sigDir)
	data, labels = scaleDownData(data, labels, 0.02)
	dataloaders = generateDataloaders(data, labels)	# 14400 ONV

	copyRed = CopyRedChannel()

	# SNN data
	offSpikes = OffSpikes()

	rate    = RateEncodeData(nSteps, 1, 0)
	latency = LatencyEncodeData(nSteps, 5, 0.01)

	# SNN labels
	spikeLabels = CopyEncodeLabels(nSteps)

	offRate    = transforms.Compose([offSpikes, rate])
	offLatency = transforms.Compose([offSpikes, latency])
	"""

	"""
	# Create npy from large csv
	csvDir = 'C:/Users/taasi/Desktop/biomechanical_eye_siggraph_asia_19/pupilData'
	npyDir = './training_data/pupil'

	data = pd.read_csv(f'{csvDir}/image_x_act_iris_sphincter.csv', header=None)
	# data = pd.read_csv(f'{csvDir}/data_smol.csv', header=None)
	data = data.to_numpy()
	np.save(f'{npyDir}/data.npy', data)

	# labels = np.genfromtxt(f'{csvDir}/output_iris_sphincter_delta_act.csv', delimiter=',')

	# np.save(f'{npyDir}/data.npy', data)
	# np.save(f'{npyDir}/labels.npy', labels)
	"""

	npyDir = './training_data/pupil'
	labels = np.load(f'./{npyDir}/labels.npy', mmap_mode='r+')
	labels = labels[:, None]
	print(labels[:10])
	subtractMean(labels, 'Pupil')
	normalization(labels, 'Pupil')



if __name__ == "__main__":
	main()