import torch
import torch.nn as nn

def getActivation(activation_type):
	"""
		Returns the various activation functions as required
	"""
	m = activation_type
	if activation_type == 'ReLU':
		return vars(nn)[m]()
	if activation_type == 'ReLU6':
		return vars(nn)[m]()
	if activation_type == 'LeakyReLU':
		return vars(nn)[m]()
	if activation_type == 'Sigmoid':
		return vars(nn)[m]()
	if activation_type == 'Tanh':
		return vars(nn)[m]()
	if activation_type == 'Softmax':
		return vars(nn)[m]()