import basicNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class HighwayFcModel(nn.Module):
	def __init__(self, inDims, input_size, output_size, numLayers, activation='ReLU', gate_activation='Sigmoid', bias = -1.0):
		super(HighwayFcModel,self).__init__()
		self.highways = nn.ModuleList([basicNet.HighwayFcNet(input_size,numLayers,activation,gate_activation) for _ in range(numLayers)])
		self.linear = nn.Linear(input_size,output_size)
		self.dimChange  = nn.Linear(inDims, input_size)

	def forward(self, x):
		x = F.relu(self.dimChange(x))
		for h in self.highways:
			x = h(x)
		x = F.softmax(self.linear(x))
		return x 


class ConvModel1D(nn.Module):
	def __init__(self, inputChannels, outputChannels, kernelWidths, usePool):
		super(ConvModel1D, self).__init()
		self.bank = [nn.ModuleList([basicNet.ConvNet1D(inputChannels, outputChannels, kernelWidths[i]) for i in range(len(kernelWidths))])]
		self.pool = nn.MaxPool1d(2, 1)
		self.usePool = usePool

	def forward(self, x):
		#input should be of the form [N,C,L]: C = input channels; L = length of sequence
		convOutput = Variable(torch.FloatTensor([]))
		for c in self.bank:
			y = c(x)
			convOutput = torch.cat((convOutput, y), -1)
		if self.usePool:
			return self.pool(convOutput)
		return convOutput
