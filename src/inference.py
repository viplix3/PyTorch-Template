from operator import mod
import os
import torch
import cv2
import numpy as np

from torch.autograd import Variable
from tqdm import tqdm
from pprint import pprint

from model import DLModel
from opts import parse_cmd_args
from utils.data_utils.data_loader import dataset as dataLoader


def test_model():
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

	if not os.path.exists(opts["model_checkpoint_dir"]):
		raise FileNotFoundError
	else:
		checkpoints = sorted(os.listdir(opts["model_checkpoint_dir"]))
		if len(checkpoints) == 0:
			print("No model checkpoints available")
			raise FileNotFoundError
		checkpoint_file = os.path.join(opts["model_checkpoint_dir"], checkpoints[-1])

	os.makedirs(opts["output_dir"], exist_ok=True)

	print("Model checkpoints will be saved\loaded from path: ", opts["model_checkpoint_dir"])
	print("Model results will be saved at path: ", opts["output_dir"], end="\n")


	dataset = dataLoader(opts, training=False).data_loader

	model = DLModel(opts)
	model.to(device)

	print("Loading model: ", checkpoint_file)
	model.load_state_dict(torch.load(checkpoint_file))
	model.eval()
	print("Model loaded")


	test_progress_bar = tqdm(len(dataset))
	file_num, num_correct_predictions = 0, 0

	test_progress_bar.set_description("[Testing %d/%d]" % (file_num, len(dataset)))
	test_progress_bar.update(file_num)

	for (image, gt_label) in dataset:
		image = Variable(image.type(Tensor))

		classifier_out = model(image)
		predicted_labels = classifier_out.max(1, keepdim=True)[1]
		num_correct_predictions += predicted_labels.eq(gt_label.to(device).view_as(predicted_labels)).sum().item()

		file_num += 1
		test_progress_bar.set_description("[Testing %d/%d] [Current Accuract: %d]" % (file_num, len(dataset),
																			(num_correct_predictions / file_num) * 100 ))
		test_progress_bar.display()
		test_progress_bar.update(file_num)
	test_progress_bar.close()

	print("Model accruacy: ", (num_correct_predictions / file_num) * 100)


if __name__ == '__main__':
	test_model()