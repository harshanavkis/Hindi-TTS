from models import HighwayFcModel, ConvModel1D
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBHG(nn.Module):
	def __init__(self, inputChannels, convSpecs, highwaySpecs, gruLen):
		"""
			-convSpecs is a list of lists containing the following:
				*filter list of required width
				*output channels
				*bool value stating whether pooling is required or not
				*the last member of the convSpecs list is to implemented separately as it doesn't pass thru ReLU
			-highwaySpecs is a list with the following specs:
				*number of fully connected layers
				*number of neurons in each layer
			-gruLen is a number stating the number of cells in the GRU layer
		"""
		super(CBHG, self).__init__()

		#convolution bank
		self.convBank = nn.ModuleList([ConvModel1D(inputChannels, convSpecs[0][1], convSpecs[0][0], convSpecs[0][2])])
		self.convBank.extend([ConvModel1D(convSpecs[i-1][1], convSpecs[i][1], convSpecs[i][0], convSpecs[i][2]) for i in range(1, len(convSpecs)-1)])
		self.lastConv = nn.Conv1d(convSpecs[-2][1], convSpecs[-1][1], convSpecs[-1][0])

		#highway bank
		self.highwayBank = HighwayFcModel(highwaySpecs[1], highwaySpecs[0]) #remember to change input dims to 128 if it doesn't match
		self.highwaySize = highwaySpecs[1]

		#GRU bank
		self.gru = nn.GRU(highwaySpecs[1], gruLen, 1, bidirectional=True)


	def forward(inp):
		x = inp
		for c in self.convBank(x):
			x = c(x)
		x = self.lastConv(x)

		#convert the input from (N,C,L) to (N,L,C)
		inp = [torch.t(i) for i in inp]
		inp = torch.stack(inp, 0)
		x = [torch.t(i) for i in x]
		x = torch.stack(x, 0)

		#residual connection
		highwayInp = inp + x
		if highwayInp.size()[2] != self.highwaySize:
			self.dimsChange = nn.Linear(highwayInp.size()[2], self.highwaySize)
			highwayInp = self.dimsChange(highwayInp)
		highwayInp = self.highwayBank(highwayInp)
		out, hidden = self.gru(highwayInp)

		return (out, hidden)
