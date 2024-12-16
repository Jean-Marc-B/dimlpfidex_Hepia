import numpy as np
import sys
from utils import output_data
filename = sys.argv[1]
outfilename = sys.argv[2]
data = np.loadtxt(filename, max_rows=100)
output_data(data, outfilename)
