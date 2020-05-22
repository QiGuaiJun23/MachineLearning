import numpy as np
import sklearn
import pandas as pd
import os
import matplotlib as mpl
import sys
import time
import tensorflow as tf
from tensorflow import keras
# print(tf.__version__)
print(sys.version_info)

for module in mpl,np,pd,sklearn,tf,keras:
	print(module.__name__,module.__version__)

