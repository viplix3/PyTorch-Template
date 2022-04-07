import os
import torch

import torchvision.transforms as transforms
from torchvision import datasets as vision_datasets

class dataset():
	def __init__(self, opts: dict, training=True) -> None:
		"""
			Assumes image dataset
			Args:
				opts (dict): _description_
				training (bool, optional): _description_. Defaults to True.
		"""
		self.dataset_dir = os.path.join(opts["data_dir"])
		os.makedirs(self.dataset_dir, exist_ok=True)

		self.transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                                        transforms.Resize((opts["inHeight"], opts["inWidth"]))
											])

		# MNIST Data used as an example
		self.dataset_obj = vision_datasets.MNIST(root=self.dataset_dir, train=training,
                                           download=True, transform=self.transforms)
		
		if training:
			self.data_loader = torch.utils.data.DataLoader(self.dataset_obj, batch_size=opts["batch_size"],
                                             shuffle=True)
		else:
			self.data_loader = torch.utils.data.DataLoader(self.dataset_obj, batch_size=1, shuffle=False)