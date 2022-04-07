import torch

from torch import nn
from typing import List
import torch.nn.functional as F

from utils import ConvBatchNorm, ConvBatchNormRelu, Downsampler


class DLModel(nn.Module):
	def __init__(self, opts: dict) -> None:
		""" Define DL model

		Args:
			opts (dict): options related to project config. Defined in opts.py
		"""
		super(DLModel, self).__init__()

		self.block_1: List[nn.Module] = []
		self.block_2: List[nn.Module] = []
		
		self.block_1.append(ConvBatchNorm(nIn=opts["model_nIn"], nOut=opts["block1_conv_nOut"], kSize=3))
		self.block_1.append(nn.MaxPool2d(kernel_size=2, stride=2))
		self.block_1.append(nn.ReLU(inplace=True))

		self.block_2.append(ConvBatchNorm(nIn=opts["block1_conv_nOut"], nOut=opts["block2_conv_nOut"], kSize=3))
		self.block_2.append(nn.MaxPool2d(kernel_size=2, stride=2))
		self.block_2.append(nn.ReLU(inplace=True))

		self.flattened_shape = opts["nIn_linear1"]
		self.fc1 = nn.Linear(in_features=self.flattened_shape, out_features=opts["nOut_linear_1"])
		self.labels_predictor = nn.Linear(in_features=opts["nOut_linear_1"], out_features=opts["numClasses"])

		self.backbone = nn.Sequential(
									*self.block_1,
									*self.block_2)

		self.predictor = nn.Sequential(
									self.fc1,
									nn.ReLU(inplace=True),
									self.labels_predictor)

	def forward(self, image: torch.Tensor) -> torch.Tensor:
		""" Forward propagates DL model on input image
		Args:
			image (torch.Tensor): image on which transformation would be applied

		Returns:
			torch.Tensor: model predictions
		"""
		features = self.backbone(image)
		features = features.view(-1, self.flattened_shape)
		model_predictions = self.predictor(features)

		model_out = F.log_softmax(model_predictions, dim=1)

		return model_out