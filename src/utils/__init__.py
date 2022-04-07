import os
import sys
import inspect

module_path = os.path.realpath(
				os.path.dirname(
					inspect.getfile(
						inspect.currentframe()
						)
					)
				)

sys.path.insert(0, module_path)

from data_utils.data_loader import *
from model_utils.layer_utils import *