import os
import numpy as np

import logging

from app_globals import *
from alad_support import *
from r_support import matrix, cbind

from forest_aad_detector import *
from results_support import write_sequential_results_to_csv

"""
To debug:
    pythonw pyalad/forest_aad_main.py
"""

logger = logging.getLogger(__name__)

dense = False  # DO NOT Change this!

args = get_command_args(debug=False)
# print "log file: %s" % args.log_file
configure_logger(args)

opts = Opts(args)
# print opts.str_opts()
logger.debug(opts.str_opts())

data = DataFrame.from_csv(opts.datafile, header=0, sep=',', index_col=None)
X_train = np.zeros(shape=(data.shape[0], data.shape[1]-1))
for i in range(X_train.shape[1]):
    X_train[:, i] = data.iloc[:, i + 1]
labels = np.array([1 if data.iloc[i, 0] == "anomaly" else 0 for i in range(data.shape[0])], dtype=int)

# X_train = X_train[0:10, :]
# labels = labels[0:10]

logger.debug("loaded file: %s" % opts.datafile)
logger.debug("results dir: %s" % opts.resultsdir)

all_num_seen = None
all_num_seen_baseline = None
all_queried_indexes = None
all_queried_indexes_baseline = None

baseline_query_info = []

opts.fid = 1  # do not change this!
runidx = 0  # do not change this!
opts.set_multi_run_options(opts.fid, runidx)

if opts.load_model and opts.modelfile != "" and os.path.isfile(opts.modelfile):
    logger.debug("Loading model from file %s" % opts.modelfile)
    mdl = load_aad_model(opts.modelfile)
else:
    rng = np.random.RandomState(args.randseed + opts.fid * opts.reruns + runidx)
    # fit the model
    mdl = AadForest(n_estimators=opts.ifor_n_trees,
                    max_samples=min(opts.ifor_n_samples, X_train.shape[0]),
                    score_type=opts.ifor_score_type, random_state=rng,
                    add_leaf_nodes_only=opts.ifor_add_leaf_nodes_only)
    mdl.fit(X_train)

logger.debug("total #nodes: %d" % (len(mdl.all_regions)))
if mdl.w is not None:
    logger.debug("w:\n%s" % str(list(mdl.w)))
else:
    logger.debug("model weights are not set")

X_train_new = mdl.transform_to_region_features(X_train, dense=dense)

w = mdl.get_uniform_weights()
agg_scores = mdl.get_score(X_train_new, w)

ensemble = Ensemble(X_train, labels, X_train_new, w,
                    agg_scores=agg_scores, original_indexes=np.arange(X_train.shape[0]),
                    auc=0.0, model=None)

metrics = mdl.aad_ensemble(ensemble, opts)

num_seen, num_seen_baseline, queried_indexes, queried_indexes_baseline = \
    summarize_ensemble_num_seen(ensemble, metrics, fid=opts.fid)
all_num_seen = rbind(all_num_seen, num_seen)
all_num_seen_baseline = rbind(all_num_seen_baseline, num_seen_baseline)
all_queried_indexes = rbind(all_queried_indexes, queried_indexes)
all_queried_indexes_baseline = rbind(all_queried_indexes_baseline, queried_indexes_baseline)
logger.debug("baseline: \n%s" % str([v for v in num_seen_baseline[0, :]]))
logger.debug("num_seen: \n%s" % str([v for v in num_seen[0, :]]))

results = SequentialResults(num_seen=all_num_seen, num_seen_baseline=all_num_seen_baseline,
                            true_queried_indexes=all_queried_indexes,
                            true_queried_indexes_baseline=all_queried_indexes_baseline)
write_sequential_results_to_csv(results, opts)

if opts.save_model:
    save_aad_model(opts.modelfile, mdl)
