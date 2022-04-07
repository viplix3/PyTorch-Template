import os
import torch

import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
from tqdm import tqdm
from pprint import pprint

from model import DLModel
from opts import parse_cmd_args
from utils.data_utils.data_loader import dataset as dataLoader


def train_model():
	opts = parse_cmd_args() # Generate arguments related to model config and training
	print("\n********** Experiment config **********")
	pprint(opts)
	print("********** Experiment config **********\n")
	
	if opts["gpu_idx"] != "-1":
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = opts["gpu_idx"]
		Tensor = torch.cuda.FloatTensor
	else:
		Tensor = torch.FloatTensor

	device = torch.device("cuda:{}".format(opts["gpu_idx"]) if torch.cuda.is_available() else "cpu")

	opts["model_checkpoint_dir"] = os.path.join(opts["model_checkpoint_dir"], opts["exp_id"], "model_weights")
	opts["output_dir"] = os.path.join(opts["output_dir"], opts["exp_id"])

	os.makedirs(opts["model_checkpoint_dir"], exist_ok=True)
	os.makedirs(opts["output_dir"], exist_ok=True)

	print("Model checkpoints will be saved\loaded from path: ", opts["model_checkpoint_dir"])
	print("Model results will be saved at path: ", opts["output_dir"], end="\n")


	dataset = dataLoader(opts).data_loader

	model = DLModel(opts)
	model.to(device)
	model.train()

	Optimizer = torch.optim.SGD(model.parameters(), lr=opts["learning_rate"])

	# torch.autograd.set_detect_anomaly(True)
	epoch_progress_bar = tqdm(range(1, opts["num_epochs"]+1))

	for epoch in epoch_progress_bar:
		epoch_progress_bar.set_description("[Epoch %d/%d]" % (epoch, opts["num_epochs"]))
		correct = 0

		for batch_idx, (image, gt_labels) in enumerate(dataset):

			image = Variable(image.type(Tensor))
			Optimizer.zero_grad() # Gradients set to zero

			classifier_out = model(image) # Forward propagation
			classifier_loss = F.nll_loss(classifier_out, gt_labels.to(device)) # Loss
			classifier_loss = torch.mean(classifier_loss) # Mean loss

			classifier_loss.backward() # Backward propagation
			Optimizer.step() # Gradient descent

			predicted_labels = classifier_out.max(1, keepdim=True)[1]
			correct += predicted_labels.eq(gt_labels.to(device).view_as(predicted_labels)).sum().item()
			current_accuracy = 100 * correct / ((batch_idx+1) * opts["batch_size"])

			epoch_progress_bar.set_description(
				"[Epoch %d/%d]  [Batch %d/%d] [Loss: %f] [Accuracy: %f]" % (epoch, opts["num_epochs"], batch_idx, len(dataset),
										classifier_loss.item(), current_accuracy))
			
		if (epoch-1) % opts["save_iter"] == 0:
			torch.save(model.state_dict(), 
							os.path.join(opts["model_checkpoint_dir"], "ModelEpoch{:04d}_{}.pth".format(epoch, datetime.now())))


if __name__ == '__main__':
	train_model()