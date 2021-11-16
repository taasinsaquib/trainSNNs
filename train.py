import time
import copy

import torch
import torch.nn as nn
from   torch.utils.tensorboard  import SummaryWriter
from   torch.optim.lr_scheduler import ReduceLROnPlateau

# *******************************************
# Training Helpers

def train_model(model, dataloaders, device, optimizer, lrScheduler, encoding, num_epochs, atol=1):
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

def test_model(model, dataloaders, device, atol, encoding):

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

def pipeline(model, dataloaders, device, epochs=25, lr=1e-2, weight_decay=0.1, encoding=False, patience=5, atol=1):

	model.to(torch.float)
	model.to(device)

	opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # TODO: tune betas
	lrSched = ReduceLROnPlateau(opt, patience=patience)
	
	best_model, m = train_model(model, dataloaders, device, opt, lrSched, encoding, num_epochs=epochs, atol=atol)

	return best_model, m

def getAtol(m, dataloaders, device, encoding=False):

	m.to(torch.float)
	m.to(device)
	m.eval()

	atols = [5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

	for atol in atols:
		print(atol)
		outputs = test_model(m, dataloaders, device, atol=atol, encoding=encoding)


def main():
	pass


if __name__ == "__main__":
	main()