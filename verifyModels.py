import numpy as np

import torch
from   torch.utils.data import Dataset, TensorDataset

import matplotlib.pyplot as plt

from models import LCN, LCNSpiking, LCNSpikingHybrid, ShallowNN, LCNChannelStack, FC
from data import ONVData


colors = {
	'LCN_normal': 'blue',
	'LCN_delta': 'deepskyblue',
	'LCNSpikingHybrid_L1_normal': 'red',
	'LCNSpikingHybrid_L1_delta': 'firebrick',
	'LCNSpikingHybrid_L2_normal': 'lightgreen',
	'LCNSpikingHybrid_L2_delta': 'green',
	'LCNSpikingHybrid_L3_normal': 'magenta',
	'LCNSpikingHybrid_L3_delta': 'purple',
	'LCNSpikingHybrid_L4_normal': 'gold',
	'LCNSpikingHybrid_L4_delta': 'darkgoldenrod',
}

# colors = {
# 	'LCN_normal': 'blue',
# 	'LCN_delta': 'navy',
# 	'LCNSpikingHybrid_L1_normal': 'red',
# 	'LCNSpikingHybrid_L1_delta': 'red',
# 	'LCNSpikingHybrid_L2_normal': 'green',
# 	'LCNSpikingHybrid_L2_delta': 'green',
# 	'LCNSpikingHybrid_L3_normal': 'purple',
# 	'LCNSpikingHybrid_L3_delta': 'purple',
# 	'LCNSpikingHybrid_L4_normal': 'orange',
# 	'LCNSpikingHybrid_L4_delta': 'orange',
# }


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


def plotAngles(testName, modelDictNames, n):

	f, axarr = plt.subplots(2, sharex=True)
	# f.suptitle(testName)

	axarr[0].set_title("Theta")
	axarr[0].set_ylabel('degrees')

	axarr[1].set_title("Phi")
	axarr[1].set_xlabel('Time')
	axarr[1].set_ylabel('degrees')

	# plot the labels
	labels = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/labels_{testName}.npy')
	labels = labels[:n]

	# n = labels.shape[0]
	# n = 306
	time = np.arange(0, n)

	xLabels = labels[:, :1]
	yLabels = labels[:, 1:]

	l, = axarr[0].plot(time, xLabels, marker='.', c='black')
	l.set_label('Actual')

	l, = axarr[1].plot(time, yLabels, marker='.', c='black')
	l.set_label('Actual')


	for i, modelDictName in enumerate(modelDictNames):

		angles = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{modelDictName}/{testName}_{modelDictName}.npy')
		angles = angles[:n]

		# X
		xAngles = angles[:, :1]

		l, = axarr[0].plot(time, xAngles, marker='.', c=colors[modelDictName])
		l.set_label(modelDictName)
		
		# Y
		yAngles = angles[:, 1:]

		l, = axarr[1].plot(time, yAngles, marker='.', c=colors[modelDictName])
		l.set_label(modelDictName)

	axarr[0].legend()
	axarr[1].legend()

	plt.show()


def plotOri(testName, modelDictNames, n, metric='Ori'):

	f, axarr = plt.subplots(2, sharex=True)
	# f.suptitle(testName)

	metricName = 'Orientation'
	units = 'degrees'

	if metric == 'Vel':
		metricName = 'Velocity'
		units = 'degrees/s'
	elif metric == 'Acc':
		metricName = 'Acceleration'
		units = 'degrees/s^2'

	axarr[0].set_title(f'Theta {metricName}')
	axarr[0].set_xlabel('Time')
	axarr[0].set_ylabel(f'{units}')

	axarr[1].set_title(f'Phi {metricName}')
	axarr[1].set_xlabel('Time')
	axarr[1].set_ylabel(f'{units}')

	time = np.arange(0, n)

	if metric == 'Ori':

		# plot the labels
		labels = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/labels_{testName}.npy')[:n]

		xLabels = labels[:, :1]
		yLabels = labels[:, 1:]

		l, = axarr[0].plot(time, xLabels, marker='.', c='black')
		l.set_label('Actual')

		l, = axarr[1].plot(time, yLabels, marker='.', c='black')
		l.set_label('Actual')

	for i, modelDictName in enumerate(modelDictNames):

		angles = np.genfromtxt(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{modelDictName}/output{metric}Leye.csv', delimiter=',')[:n]

		# X
		xAngles = angles[:, 1:2]

		l, = axarr[0].plot(time, xAngles, marker='.', c=colors[modelDictName])
		l.set_label(modelDictName)
		
		# Y
		yAngles = angles[:, 0:1]

		l, = axarr[1].plot(time, yAngles, marker='.', c=colors[modelDictName])
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

	# Do once
	testName = 'smooth_lateral'
	convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels/smooth_lateral', 'actual_smooth_lateral.csv', f'labels_{testName}.npy')

	testName = 'smooth'
	convertCSVtoNpy('C:/Users/taasi/Desktop/trainSNNs/verifyModels/smooth', 'actual_smooth.csv', f'labels_{testName}.npy')

	testName = 'saccade'
	convertCSVtoNpy(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}', 'actual_smooth.csv', f'labels_{testName}.npy')

	testName = 'projectile_forward'
	convertCSVtoNpy(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}', 'actual_smooth.csv', f'labels_{testName}.npy')

	testName = 'projectile_sideways'
	convertCSVtoNpy(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}', 'actual_smooth.csv', f'labels_{testName}.npy')

	# SMOOTH LATERAL
	"""
	testName = 'smooth_lateral'

	models = ['FC_normal', 'LCN_normal', 'LCN_normal_lr', 'LCN_delta', 'LCN_delta_lr']
	models += ['LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L2_normal', 'LCNSpikingHybrid_L2_delta']
	models += ['LCNSpikingHybrid_L3_delta', 'LCNSpikingHybrid_L3_normal', 'LCNSpikingHybrid_L4_normal', 'LCNSpikingHybrid_L4_delta']

	for m in models:
		folder = f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{m}'
		convertCSVtoNpy(folder, 'nn_out.csv', f'{testName}_{m}.npy')

	# TODO: pass colors dict
	# plotAngles(testName, ['FC_normal', 'LCN_normal', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L4_normal'], 400)
	# plotAngles(testName, ['LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L4_normal'], 400)
	# plotAngles(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L4_delta'], 400)


	plotAngles(testName, ['FC_normal', 'LCN_normal'], 400)

	plotAngles(testName, ['LCN_normal', 'LCN_delta'], 400)

	plotAngles(testName, ['LCN_normal', 'LCN_normal_lr'], 400)
	plotAngles(testName, ['LCN_delta', 'LCN_delta_lr'], 400)

	plotAngles(testName, ['LCN_normal', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L2_normal', 'LCNSpikingHybrid_L3_normal', 'LCNSpikingHybrid_L4_normal'], 400)
	plotAngles(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L2_delta', 'LCNSpikingHybrid_L3_delta', 'LCNSpikingHybrid_L4_delta'], 400)
	"""

	# SMOOTH
	"""
	testName = 'smooth'

	models = ['FC_normal', 'LCN_normal', 'LCN_delta']
	models += ['LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L2_normal', 'LCNSpikingHybrid_L2_delta']
	models += ['LCNSpikingHybrid_L3_delta', 'LCNSpikingHybrid_L3_normal', 'LCNSpikingHybrid_L4_normal', 'LCNSpikingHybrid_L4_delta']

	for m in models:
		folder = f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{m}'
		convertCSVtoNpy(folder, 'nn_out.csv', f'{testName}_{m}.npy')

	# plotAngles(testName, ['LCN_normal', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L4_normal'], 470)
	# plotAngles(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L4_delta'], 470)
	# # plotAngles(testName, ['LCNSpikingHybrid_L4_normal', 'LCNSpikingHybrid_L4_normal_gain2'], 370)

	# plotAngles(testName, ['LCN_normal', 'LCN_delta'], 470)

	# plotOri(testName, ['LCN_normal', 'LCN_delta'], 470)

	# plotAngles(testName, ['LCN_normal', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L2_normal', 'LCNSpikingHybrid_L3_normal', 'LCNSpikingHybrid_L4_normal'], 470)
	# plotAngles(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L2_delta', 'LCNSpikingHybrid_L3_delta', 'LCNSpikingHybrid_L4_delta'], 470)

	# plotOri(testName, ['LCN_normal', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L2_normal', 'LCNSpikingHybrid_L3_normal', 'LCNSpikingHybrid_L4_normal'], 470)
	# plotOri(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L2_delta', 'LCNSpikingHybrid_L3_delta', 'LCNSpikingHybrid_L4_delta'], 470)
	"""

	# SACCADE
	"""
	testName = 'saccade'

	models = ['LCN_normal', 'LCN_delta']
	models += ['LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L2_normal', 'LCNSpikingHybrid_L2_delta']
	models += ['LCNSpikingHybrid_L3_delta', 'LCNSpikingHybrid_L3_normal', 'LCNSpikingHybrid_L4_normal', 'LCNSpikingHybrid_L4_delta']

	for m in models:
		folder = f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{m}'
		convertCSVtoNpy(folder, 'nn_out.csv', f'{testName}_{m}.npy')


	# plotAngles(testName, ['LCN_normal', 'LCN_delta'], 185)

	# plotOri(testName, ['LCN_normal', 'LCN_delta'], 185)

	plotAngles(testName, ['LCN_normal', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L2_normal', 'LCNSpikingHybrid_L3_normal', 'LCNSpikingHybrid_L4_normal'], 185)
	plotAngles(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L2_delta', 'LCNSpikingHybrid_L3_delta', 'LCNSpikingHybrid_L4_delta'], 185)
	plotAngles(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L2_delta', 'LCNSpikingHybrid_L3_delta', 'LCNSpikingHybrid_L4_delta'], 185)

	plotOri(testName, ['LCN_normal', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L2_normal', 'LCNSpikingHybrid_L3_normal', 'LCNSpikingHybrid_L4_normal'], 185)
	plotOri(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L2_delta', 'LCNSpikingHybrid_L3_delta', 'LCNSpikingHybrid_L4_delta'], 185)

	plotOri(testName, ['LCN_normal', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L4_normal'], 185, 'Vel')
	plotOri(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L4_delta'], 185, 'Vel')

	plotOri(testName, ['LCN_normal', 'LCNSpikingHybrid_L1_normal', 'LCNSpikingHybrid_L4_normal'], 185, 'Acc')
	plotOri(testName, ['LCN_delta', 'LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L4_delta'], 185, 'Acc')
	"""


	# PROJECTILE FORWARD
	""" 
	testName = 'projectile_forward'

	models = ['LCNSpikingHybrid_L2_delta']

	for m in models:
		folder = f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{m}'
		convertCSVtoNpy(folder, 'nn_out.csv', f'{testName}_{m}.npy')

	plotAngles(testName, ['LCNSpikingHybrid_L2_delta'], 80)


	# PROJECTILE SIDEWAYS
	testName = 'projectile_sideways'

	models = ['LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L2_delta', 'LCNSpikingHybrid_L4_delta']

	for m in models:
		folder = f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{m}'
		convertCSVtoNpy(folder, 'nn_out.csv', f'{testName}_{m}.npy')

	plotAngles(testName, ['LCNSpikingHybrid_L1_delta', 'LCNSpikingHybrid_L2_delta', 'LCNSpikingHybrid_L4_delta'], 80)
	"""


	# Compare Eye Motions to Real Eye
	"""
	testName = 'saccade'
	files = ['Ori', 'Vel', 'Acc']

	for f in files:
		convertCSVtoNpy(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}', f'output{f}Leye.csv', f'output{f}Leye.npy')

	# 667 
	n = 274

	t = np.arange(n)

	ori = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/outputOriLeye.npy')[:n]
	vel = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/outputVelLeye.npy')[:n]
	acc = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/outputAccLeye.npy')[:n]

	plt.plot(t, ori)
	plt.show()

	plt.plot(t, vel)
	plt.show()

	plt.plot(t, acc)
	plt.show()

	model = 'LCN_normal'
	for f in files:
		convertCSVtoNpy(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{model}', f'output{f}Leye.csv', f'output{f}Leye.npy')

	model = 'LCNHybrid_L4_delta'
	ori = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{model}/outputOriLeye.npy')[:n]
	vel = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{model}/outputVelLeye.npy')[:n]
	acc = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{model}/outputAccLeye.npy')[:n]

	model = 'LCN_normal'
	ori2 = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{model}/outputOriLeye.npy')[:n]
	vel2 = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{model}/outputVelLeye.npy')[:n]
	acc2 = np.load(f'C:/Users/taasi/Desktop/trainSNNs/verifyModels/{testName}/{model}/outputAccLeye.npy')[:n]

	plt.subplot(1, 2, 1)
	plt.plot(t, ori)
	plt.subplot(1, 2, 2)
	plt.plot(t, ori2)
	plt.show()

	plt.subplot(1, 2, 1)
	plt.plot(t, vel)
	plt.subplot(1, 2, 2)
	plt.plot(t, vel2)
	plt.show()

	plt.subplot(1, 2, 1)
	plt.plot(t, acc)
	plt.subplot(1, 2, 2)
	plt.plot(t, acc2)
	
	plt.show()
	"""

	colorONV = np.genfromtxt(f'C:/Users/taasi/Desktop/image_x.csv', delimiter=',')
	
	i = 0

	n = 14400

	x = colorONV[0, :n]
	y = colorONV[0, n:2*n]
	z = colorONV[0, 2*n:3*n]

	print((x==y).all())
	print((y==z).all())
	print((x==z).all())


if __name__ == '__main__':
	main()
