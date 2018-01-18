import basicNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayFcModel(nn.Module):
	def __init__(self, inDims, input_size, output_size, numLayers, activation='ReLU', gate_activation='Sigmoid', bias = -1.0):
		super(HighwayFcModel,self).__init__()
		self.highways = nn.ModuleList([basicNet.HighwayFcNet(input_size,numLayers,activation,gate_activation) for _ in range(numLayers)])
		self.linear = nn.Linear(input_size,output_size)
		self.dimChange  = nn.Linear(inDims, input_size)

	def forward(self,x):
		x = F.relu(self.dimChange(x))
		for h in self.highways:
			x = h(x)
		x = F.softmax(self.linear(x))
		return x 

