import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from activations import getActivation as gA

class HighwayFcNet(nn.Module):
	"""
		A more robust fully connected network
		return: H*T + (1-T)x
	"""
	def __init__(self, input_size, numLayers, activation_type='ReLU',gate_activation='Sigmoid',bias=-1.0): #activation_type is a string containing the name of the activation
		"""
			We create a group of highway fc layers
			All layers have the same number of units
			Different number of units can be achieved through Plain Fully connected layers
		"""
		super(HighwayFcNet,self).__init__()
		self.activation = gA(activation_type) #H func
		self.gate_activation = gA(gate_activation)#T func
		self.plain = nn.Linear(input_size,input_size)
		nn.init.xavier_uniform(self.plain.weight)
		self.gate = nn.Linear(input_size,input_size)
		self.gate.bias.data.fill_(bias)

	def forward(self,x):
		h_out = self.activation(self.plain(x))
		t_out = self.gate_activation(self.gate(x))
		return torch.add(torch.mul(h_out,t_out),torch.mul((1.0-t_out),x))

