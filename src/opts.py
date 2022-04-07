import argparse
import os

from easydict import EasyDict as edict

def parse_cmd_args():
	args = argparse.ArgumentParser(description="CLI arguments")

	# Project directory setup
	args.add_argument("--data_dir", type=str, default="../resources/data/",
						required=False, help="Input data path")
	args.add_argument("--output_dir", type=str, default="../model_out/",
						required=False, help="Model output dumping path")
	args.add_argument("--model_checkpoint_dir", type=str, default="../resources/",
						required=False, help="Model checkpoint path")

	# Code execution related params
	args.add_argument("--inHeight", type=int, default=32,
						required=False, help="Input image height")
	args.add_argument("--inWidth", type=int, default=32,
						required=False, help="Input image width")
	args.add_argument("--gpu_idx", type=str, default="0",
						required=False, help="GPU index to be used")
	
	# Model training config
	args.add_argument("--exp_id", type=str, default="test_implementation",
						required=False, help="Experiment name for saving model outputs (checkpoints)")
	args.add_argument("--num_epochs", type=int, default=6,
						required=False, help="Number of epochs for model training")
	args.add_argument("--learning_rate", type=int, default=1e-3,
						required=False, help="Learning rate for model training")
	args.add_argument("--batch_size", type=int, default=64,
						required=False, help="Batch size for model training")
	args.add_argument("--save_iter", type=int, default=1,
						required=False, help="Number of epochs after which the model weights should be saved")
	


	# Parse all arguments as a dict
	opts = edict(vars(args.parse_args()))

	# Model config
	opts["down_ratio"] = 4 # Network downsampling factor
	opts["model_nIn"] = 1 # Input image channels

	opts["block1_conv_nOut"] = 32 # Block-1 convolution output channels
	opts["block2_conv_nOut"] = 64 # Block-2 convolution output channels

	opts["nIn_linear1"] = int((opts["inHeight"] / opts["down_ratio"]) * (opts["inWidth"] / opts["down_ratio"]) \
							* opts["block2_conv_nOut"]) # FC-1 layer input
	opts["nOut_linear_1"] = 128 # FC-2 layer output
	opts["nOut_linear_2"] = 64
	opts["numClasses"] = 10

	return opts