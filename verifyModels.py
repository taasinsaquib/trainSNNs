import numpy as np

import torch
from   torch.utils.data import Dataset, TensorDataset

import matplotlib.pyplot as plt

from models import LCN, LCNSpiking, LCNSpikingHybrid, ShallowNN, LCNChannelStack, FC
from data import ONVData
#
# Compare two ONV files I have and see if they're the same
# TODO: run FC_200epoch on collected ONVs, write ONVs with simulation to file and compare
# THen I can test without the siggraph code


def compareONVs():
	# compare two recordings of ONVs

	onv1 = 'C:/Users/taasi/Desktop/trainSNNs/verifyModels/image_x.csv'
	onv2 = 'C:/Users/taasi/Desktop/biomechanical_eye_siggraph_asia_19/perceptionData/image_x.csv'

	onv1 = np.genfromtxt(onv1, delimiter=',')
	onv2 = np.genfromtxt(onv2, delimiter=',')

	print(onv1.shape, onv2.shape)

	onv1 = onv1[:1]
	onv2 = onv2[:1]

	same = (onv1 == onv2).all()
	print(same)


def convertCSVtoNpy(path, csvFile, npyFile):

	data = np.genfromtxt(f'{path}/{csvFile}', delimiter=',')
	# data = data[:10]
	np.save(f'{path}/{npyFile}', data)


def createDataLoader(dir, dataFile, labelFile, xTransform=None, yTransform=None, numWorkers=1, batchSize=8):
	data = np.load(f'{dir}/{dataFile}', mmap_mode='r+')
	labels = np.load(f'{dir}/{labelFile}', mmap_mode='r+')

	print(f'Data shape: {data.shape}, Labels shape: {labels.shape}')

	X_data_tensor = torch.from_numpy(data).float()
	y_data_tensor = torch.from_numpy(labels).float()
	init_dataset = TensorDataset(X_data_tensor, y_data_tensor)

	test_data = ONVData(init_dataset,  xTransform=xTransform, yTransform=yTransform)

	dataloaders = {
		'test': torch.utils.data.DataLoader(test_data,  batch_size=batchSize, shuffle=False, num_workers=numWorkers),
	}

	return dataloaders


def testModel(model, dataloaders, device, encoding):

	angles = None

	for inputs, labels in dataloaders['test']:

		# if encoding == True:
		# 	labels = labels.permute((1, 0, 2))

		inputs = inputs.float()
		# labels = labels.float()

		inputs = inputs.to(device)
		# labels = labels.to(device)

		if encoding == True:
			mem, spikes, outputs = model(inputs)
		else:
			outputs = model(inputs)

		# print("outputs")
		# print(outputs.cpu().detach().numpy()[:5])
		
		# print("labels")
		# print(labels.cpu().detach().numpy()[:5])

		# print("huh", outputs[-1], labels[-1])

		outputs = outputs.cpu().detach().numpy()

		if angles is None:
			angles = outputs
		else:
			angles = np.vstack((angles, outputs))

	# convert to degrees
	angles *= (180/np.pi)
	return angles


def runTest(testName, modelDictName, modelType, device):

	dataloader = createDataLoader('C:/Users/taasi/Desktop/trainSNNs/verifyModels/', 'dataTest.npy', 'labelsTest.npy')

	if modelType == 'FC':
		m = FC()
	elif modelType == "LCN":
		m = LCN(14400, 2, 25, 2, 5, True)

	modelDictPath = 'C:/Users/taasi/Desktop/trainSNNs/model_dicts'
	m.load_state_dict(torch.load(f'{modelDictPath}/{modelDictName}', map_location=device))
	m.to(torch.float)
	m.to(device)

	angles = testModel(m, dataloader, device, encoding=False)
	np.save(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}_{modelDictName}.npy', angles)

def plotAngles(testName, modelDictNames):

	f, axarr = plt.subplots(2, sharex=True)

	f.suptitle(testName)

	axarr[0].set_title("Compare Theta")
	axarr[0].set_ylabel('Theta (degrees)')

	axarr[1].set_title("Compare Phi")
	axarr[1].set_xlabel('Time')
	axarr[1].set_ylabel('Phi (degrees)')

	# plot the labels
	labels = np.load('C:/Users/taasi/Desktop/trainSNNs/verifyModels/labelsTest.npy')
	labels = labels[:306]

	n = labels.shape[0]
	n = 306
	time = np.arange(0, n)

	xLabels = labels[:, :1]
	yLabels = labels[:, 1:]

	l, = axarr[0].plot(time, xLabels, marker='.')
	l.set_label('Actual')

	l, = axarr[1].plot(time, yLabels, marker='.')
	l.set_label('Actual')


	for i, modelDictName in enumerate(modelDictNames):

		angles = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}_{modelDictName}.npy')
		angles = angles[:306]

		# X
		xAngles = angles[:, :1]

		l, = axarr[0].plot(time, xAngles, marker='.')
		l.set_label(modelDictName)
		
		# Y
		yAngles = angles[:, 1:]

		l, = axarr[1].plot(time, yAngles, marker='.')
		l.set_label(modelDictName)

	axarr[0].legend()
	axarr[1].legend()

	plt.show()


# channelstackspiking
def main():
	
	"""
	# running ONV here vs through simulation is different for some reason
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	# device = 'cpu'

	# compareONVs() 

	# Convert CSV to NPY once (for data (ONV) and labels (angles))
	# convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels/', 'image_x.csv', 'dataTest.npy')
	convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels/', 'smooth_lateral_actual.csv', 'labelsTest.npy')

	# print(onv.shape, angles.shape)

	# runTest('smoothLateral', 'FC_normal_100epoch', 'FC', device) 
	# runTest('smoothLateral', 'FC_normal_200epoch', 'FC', device) 

	plotAngles('smoothLateral', ['FC_normal_100epoch', 'FC_normal_200epoch', 'hi'])
	"""

	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels/', 'image_x.csv', 'dataTest.npy')
	# convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels/', 'actual_smooth_lateral.csv', 'labelsTest.npy')

	# runTest('smoothLateral', 'LCN_normal_100epoch_k25', 'LCN', device) 

	# convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels/', 'actual_smooth_lateral.csv', 'labelsTest.npy')
	# convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels', 'smooth_lateral_LCN_normal_100epoch_k25.csv', 'smoothLateral_actual_LCN_normal_100epoch_k25.npy')
	plotAngles('smoothLateral', ['actual_LCN_normal_100epoch_k25', 'LCN_normal_100epoch_k25'])
	"""

	convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels/', 'actual_smooth_lateral.csv', 'labelsTest.npy')
	convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels', 'smooth_LCNSpikingHybrid_delta_100epoch_k25_L4.csv', 'smooth_actual_LCNSpikingHybrid_delta_100epoch_k25_L4.npy')
	plotAngles('smooth', ['actual_LCNSpikingHybrid_delta_100epoch_k25_L4'])


if __name__ == '__main__':
	main()


# TODO
# collect onv for smooth lateral
	# compare values here to values printed