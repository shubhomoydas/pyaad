import numpy as np
from numpy import random

import logging

from app_globals import *

from random_split_trees import *
from forest_aad_detector import *

logger = logging.getLogger(__name__)

args = get_command_args(debug=False)
# print "log file: %s" % args.log_file
configure_logger(args)

"""
python pyalad/test_random_split_trees.py --log_file=./temp/random_split_trees.log --debug
"""

# X = np.zeros((5, 10), dtype=float)
rnd = np.random.RandomState(args.randseed)
X = rnd.uniform(0, 1, (10, 5))
logger.debug("X:\n%s" % str(X))

X_new = rnd.uniform(0, 1, (5, 5))
logger.debug("X_new:\n%s" % str(X_new))

starttime = timer()
if False:
    mdl_file = "./temp/random_split_trees.mdl"
    if True:
        hst = HSTree(max_depth=4, max_features=X.shape[1])
        hst.fit(X, None)
    else:
        hst = load_aad_model(mdl_file)

    logger.debug("\n%s" % str(hst.tree_))
    logger.debug("Node samples:\n%s" % str(hst.tree_.n_node_samples))

    save_aad_model(mdl_file, hst)
elif True:
    hst = HSTrees(n_estimators=1, max_depth=4, max_features=X.shape[1],
                  n_jobs=4, random_state=rnd)
    hst.fit(X, y=None)
    leaves, nodeinds = hst.estimators_[0].tree_.apply(X, getleaves=True, getnodeinds=True)
    leaves = np.unique(leaves)
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
    leaves_new, nodeinds_new = hst.estimators_[0].tree_.apply(X_new, getleaves=True, getnodeinds=True)
    leaves_new = np.unique(leaves_new)
    logger.debug("leaves:\n%s" % str(list(leaves_new)))
    hst.add_samples(X_new, current=False)
    nsamples = hst.estimators_[0].tree_.n_node_samples[leaves_new]
    nsamples_buffer = hst.estimators_[0].tree_.n_node_samples_buffer[leaves_new]
    logger.debug("Node samples after add:\n%s\n%s" % (str(list(nsamples)), str(list(nsamples_buffer))))
    hst.update_model_from_stream_buffer()
    nsamples = hst.estimators_[0].tree_.n_node_samples[leaves_new]
    nsamples_buffer = hst.estimators_[0].tree_.n_node_samples_buffer[leaves_new]
    logger.debug("Node samples after move:\n%s\n%s" % (str(list(nsamples)), str(list(nsamples_buffer))))
elif False:
    X[:, 4] = 1.
    X[0, 4] = 0.
    sp = RSForestSplitter()
    mn, mx = sp.get_feature_ranges(X)
    logger.debug("X:\n%s" % str(X))
    logger.debug("mn:\n%s\nmx:\n%s" % (str(list(mn)), str(list(mx))))
elif False:
    rst = RSTree(max_depth=1, max_features=X.shape[1], random_state=rnd)
    rst.fit(X, None)
    logger.debug("\n%s" % str(rst.tree_))
    logger.debug("Node samples:\n%s" % str(rst.tree_.n_node_samples))
else:
    rst = RSTree(max_depth=1, max_features=X.shape[1], random_state=rnd)
    rst.fit(X, None)
    logger.debug("\n%s" % str(rst.tree_))
    logger.debug("Node samples:\n%s" % str(rst.tree_.n_node_samples))
    rst.tree_.add_samples(X_new, current=False)
    logger.debug("Node samples after add:\n%s" % str(rst.tree_.n_node_samples))
    rst.tree_.update_model_from_stream_buffer()
    logger.debug("Node samples after move:\n%s" % str(rst.tree_.n_node_samples))

endtime = timer()
tdiff = difftime(endtime, starttime, units="secs")
logger.debug("Completed in %f sec(s)" % (tdiff))

logger.debug("test completed...")
