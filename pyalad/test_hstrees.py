import numpy as np
from numpy import random

import logging
import cPickle
import gzip

from app_globals import *

from HSTrees import *

logger = logging.getLogger(__name__)

args = get_command_args(debug=False)
# print "log file: %s" % args.log_file
configure_logger(args)

"""
python pyalad/test_hstrees.py --log_file=./temp/hstrees.log --debug
"""

def save_aad_model(filepath, model):
    f = gzip.open(filepath, 'wb')
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def load_aad_model(filepath):
    f = gzip.open(filepath, 'rb')
    model = cPickle.load(f)
    f.close()
    return model


# X = np.zeros((5, 10), dtype=float)
rnd = np.random.RandomState(args.randseed)
X = rnd.uniform(0, 1, (10, 5))
logger.debug("X:\n%s" % str(X))

starttime = timer()
if False:
    mdl_file = "/Users/moy/work/git/pyalad/temp/hstrees.mdl"
    if False:
        hst = HSTree(None, HSSplitter(), 4, X.shape[1], None)
        hst.fit(X, None)
    else:
        hst = load_aad_model(mdl_file)

    logger.debug("\n%s" % str(hst.tree_))
    logger.debug("Node samples:\n%s" % str(hst.tree_.n_node_samples))

    save_aad_model(mdl_file, hst)
else:
    hst = HSTrees(n_estimators=10, max_depth=4, n_jobs=4, random_state=rnd)
    hst.fit(X, None, None)
    leaves, nodeinds = hst.estimators_[0].tree_.apply(X, getnodeinds=True)
    logger.debug("leaves:\n%s" % str(list(leaves)))
    if nodeinds is not None:
        logger.debug("node inds:\n%s" % str(nodeinds.toarray()))
        depths = nodeinds.sum(axis=1)
        logger.debug("depths:\n%s" % str(list(np.transpose(depths))))
        nsamples = hst.estimators_[0].tree_.n_node_samples[leaves]
        logger.debug("node samples:\n%s" % str(list(nsamples)))
    scores = hst.estimators_[0].decision_function(X)
    logger.debug("scores:\n%s" % str(list(scores)))

    scores = hst.decision_function(X)
    logger.debug("forest scores:\n%s" % str(list(scores)))

endtime = timer()
tdiff = difftime(endtime, starttime, units="secs")
logger.debug("Completed in %f sec(s)" % (tdiff))

logger.debug("test completed...")
