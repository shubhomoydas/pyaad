import numpy as np
import scipy as sp

from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix, vstack

import logging
from app_globals import *

from r_support import *

from data_stream import *

"""
python pyalad/test_stream.py --log_file=./temp/stream.log --debug
"""

logger = logging.getLogger(__name__)

args = get_command_args(debug=False)
# print "log file: %s" % args.log_file
configure_logger(args)

rnd = np.random.RandomState(args.randseed)
X = rnd.uniform(0, 1, (10, 5))
logger.debug("X:\n%s" % str(X))

X_new = rnd.uniform(0, 1, (5, 5))
logger.debug("X_new:\n%s" % str(X_new))

tm = Timer()
tm.start()
ds = DataStream(X)
i = 0
while not ds.empty():
    i += 1
    insts = ds.read_next_from_stream(2)
    logger.debug("%03d:\n%s" % (i, str(insts)))
tm.message("Completed in")
logger.debug("test completed...")
