import os
import numpy as np

import logging

from app_globals import *
from alad_support import *
from r_support import matrix, cbind

from alad_iforest import *
from results_support import write_sequential_results_to_csv
from isolation_forest_support import iforest_unit_tests_battery, \
    get_original_iforest_results, get_queried_indexes, write_baseline_query_indexes

"""
To debug:
    pythonw pyalad/test_simple_iforest_ensemble.py
"""

logger = logging.getLogger(__name__)


def get_tree_scores(samples, mdl):
    n_samples = samples.shape[0]
    # n_samples_leaf = np.zeros((n_samples, mdl.clf.n_estimators))
    depths = np.zeros((n_samples, mdl.clf.n_estimators), order="f")
    for i, tree in enumerate(mdl.clf.estimators_):
        leaves_index = tree.apply(samples)
        node_indicator = tree.decision_path(samples)
        # n_samples_leaf[:, i] = tree.tree_.n_node_samples[leaves_index]
        depths[:, i] = np.asarray(node_indicator.sum(axis=1)).reshape(-1) - 1
    return -depths


if True:
    # PRODUCTION code
    args = get_command_args(debug=False)
    # print "log file: %s" % args.log_file
    configure_logger(args)
else:
    # DEBUG code
    datasets = ["abalone", "ann_thyroid_1v3", "cardiotocography_1", "covtype_sub",
                "kddcup_sub", "mammography_sub", "shuttle_sub", "yeast", "toy", "toy2"]

    dataset = datasets[8]
    datapath = "/Users/moy/work/datasets/anomaly/%s/fullsamples/%s_1.csv" % (dataset, dataset)
    outputdir = "/Users/moy/work/temp/aad_iforest"

    budget = 60  # 10
    n_runs = 10  # 10
    n_trees = 100
    n_samples = 256
    Cx = 1000
    Ca = 100
    inference_type = AAD_UPD_TYPE
    args = get_aad_iforest_args(dataset=dataset, inference_type=inference_type,
                                n_trees=n_trees, n_samples=n_samples,
                                budget=budget, reruns=n_runs,
                                log_file="/Users/moy/work/temp/aad_iforest/aad_iforest.txt")
    args.datafile = datapath
    args.resultsdir = os.path.join(outputdir, args.dataset, "if_aad_%d_%d_%d_sig%4.3f_cx%4.3f" %
                                   (args.ifor_n_trees, args.ifor_n_samples, args.budget,
                                    args.sigma2, args.Cx))
    dir_create(args.resultsdir)

opts = Opts(args)
# print opts.str_opts()
logger.debug(opts.str_opts())

data = DataFrame.from_csv(opts.datafile, header=0, sep=',', index_col=None)
X_train = np.zeros(shape=(data.shape[0], data.shape[1]-1))
for i in range(X_train.shape[1]):
    X_train[:, i] = data.iloc[:, i + 1]
labels = np.array([1 if data.iloc[i, 0] == "anomaly" else 0 for i in range(data.shape[0])], dtype=int)

# X_train = X_train[0:40, :]
# labels = labels[0:40]

logger.debug("loaded file: %s" % opts.datafile)
logger.debug("results dir: %s" % opts.resultsdir)

mdl = None


opts.fid = 1

all_num_seen = None
all_num_seen_baseline = None
all_queried_indexes = None
all_queried_indexes_baseline = None

for runidx in opts.get_runidxs():
    opts.set_multi_run_options(opts.fid, runidx)

    rng = np.random.RandomState(args.randseed + opts.fid * opts.reruns + runidx)

    # fit the model
    mdl = AadIsolationForest(n_estimators=opts.ifor_n_trees,
                             max_samples=min(opts.ifor_n_samples, X_train.shape[0]),
                             score_type=opts.ifor_score_type, random_state=rng,
                             add_leaf_nodes_only=opts.ifor_add_leaf_nodes_only)
    mdl.fit(X_train)
    logger.debug("total #nodes: %d" % (len(mdl.all_regions)))

    tree_scores = get_tree_scores(X_train, mdl)
    # logger.debug(tree_scores)

    rnd_seed = opts.randseed + runidx
    num_seen_summary = run_alad_simple(X_train, labels, opts,
                                       scores=tree_scores,
                                       rnd_seed=rnd_seed)

    all_num_seen = rbind(all_num_seen, num_seen_summary[0])
    all_num_seen_baseline = rbind(all_num_seen_baseline, num_seen_summary[1])
    all_queried_indexes = rbind(all_queried_indexes, num_seen_summary[2])
    all_queried_indexes_baseline = rbind(all_queried_indexes_baseline, num_seen_summary[3])

    logger.debug("completed %s runid %d" % (opts.dataset, runidx,))

results = SequentialResults(num_seen=all_num_seen, num_seen_baseline=all_num_seen_baseline,
                            true_queried_indexes=all_queried_indexes,
                            true_queried_indexes_baseline=all_queried_indexes_baseline)

opts.fid = 0
opts.runidx = 0
write_sequential_results_to_csv(results, opts)

print "completed alad %s for %s" % (opts.detector_type_str(), opts.dataset,)
