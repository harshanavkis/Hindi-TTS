import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

"""
	To be used for the encoder and decoder pre-nets 
"""

class FcNet(nn.Module):
	def __init__(self, inputSize, outputSize, dropout = 0.5):
		super(FcNet, self).__init__()
		self.dropoutval = dropout
		self.fc = nn.Linear(inputSize, outputSize)
		self.dropout = nn.Dropout(self.dropoutval)

	def forward(self, x):
		x = self.fc(x)
		x = F.relu(x)
		x = self.dropout(x)
		return x

class preNetModel(nn.Module):
	"""
		* dimArr is a list of the hidden dimensions of the fully connected net
		* implemented in this manner to provide a more robust functionality
	"""
	def __init__(self, inputSize, dimArr, dropout = 0.5):
		super(preNetModel, self).__init__()
		self.linears = nn.ModuleList([FcNet(inputSize, dimArr[0], dropout)])
		self.linears.extend([FcNet(dimArr[i-1], dimArr[i]) for i in range(1, len(dimArr))])

	def forward(self, x):
		for l in self.linears:
			x = l(x)
		return x