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
    pythonw pyalad/isolation_forest_detector.py
"""

logger = logging.getLogger(__name__)

dense = False  # DO NOT Change this!

args = get_command_args(debug=False)
# print "log file: %s" % args.log_file
configure_logger(args)

opts = Opts(args)
# print opts.str_opts()
logger.debug(opts.str_opts())

run_aad = True
run_tests = False and opts.reruns == 1

run_orig_iforest_only = opts.update_type == IFOREST_ORIG

baseline_query_indexes_only = False

data = DataFrame.from_csv(opts.datafile, header=0, sep=',', index_col=None)
X_train = np.zeros(shape=(data.shape[0], data.shape[1]-1))
for i in range(X_train.shape[1]):
    X_train[:, i] = data.iloc[:, i + 1]
labels = np.array([1 if data.iloc[i, 0] == "anomaly" else 0 for i in range(data.shape[0])], dtype=int)

# X_train = X_train[0:10, :]
# labels = labels[0:10]

logger.debug("loaded file: %s" % opts.datafile)
logger.debug("results dir: %s" % opts.resultsdir)

mdl = None
X_train_new = None
metrics = None
if run_aad:
    # use this to run AAD

    opts.fid = 1

    all_num_seen = None
    all_num_seen_baseline = None
    all_queried_indexes = None
    all_queried_indexes_baseline = None

    all_baseline = ""
    all_orig_iforest = ""

    all_orig_num_seen = None

    baseline_query_info = []

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

        if run_orig_iforest_only:
            orig_num_seen = get_original_iforest_results(X_train, labels, mdl, opts.fid, runidx, opts)
            all_orig_num_seen = rbind(all_orig_num_seen, orig_num_seen)
            continue

        X_train_new = mdl.transform_to_region_features(X_train, dense=dense)

        w = np.ones(len(mdl.d), dtype=float)
        w = w / w.dot(w)  # normalized uniform weights

        agg_scores = mdl.get_score(X_train_new, w)

        if baseline_query_indexes_only:
            baseline_query_info.append(get_queried_indexes(agg_scores, labels, opts))
            continue

        ensemble = Ensemble(X_train, labels, X_train_new, w,
                            agg_scores=agg_scores, original_indexes=np.arange(X_train.shape[0]),
                            auc=0.0, model=None)

        if False:
            metrics = alad_ensemble(ensemble, opts)
        else:
            metrics = mdl.aad_ensemble(ensemble, opts)

        if metrics is not None:
            num_seen, num_seen_baseline, queried_indexes, queried_indexes_baseline = \
                summarize_ensemble_num_seen(ensemble, metrics, fid=opts.fid)
            all_num_seen = rbind(all_num_seen, num_seen)
            all_num_seen_baseline = rbind(all_num_seen_baseline, num_seen_baseline)
            all_queried_indexes = rbind(all_queried_indexes, queried_indexes)
            all_queried_indexes_baseline = rbind(all_queried_indexes_baseline, queried_indexes_baseline)
            logger.debug("baseline: \n%s" % str([v for v in num_seen_baseline[0, :]]))
            logger.debug("num_seen: \n%s" % str([v for v in num_seen[0, :]]))
        else:
            queried = np.argsort(-agg_scores)
            n_found = np.cumsum(labels[queried[np.arange(60)]])
            all_baseline = all_baseline + ",".join([str(v) for v in n_found]) + os.linesep

            orig_iforest_scores = mdl.decision_function(X_train)  # smaller is more anomalous
            queried = np.argsort(orig_iforest_scores)
            n_found = np.cumsum(labels[queried[np.arange(60)]])
            all_orig_iforest = all_orig_iforest + ",".join([str(v) for v in n_found]) + os.linesep
            logger.debug("Completed runidx: %d" % runidx)

        if not run_tests:
            metrics = None  # release memory
            mdl = None
            X_train_new = None
            ensemble = None

    if all_num_seen is not None:
        results = SequentialResults(num_seen=all_num_seen, num_seen_baseline=all_num_seen_baseline,
                                    true_queried_indexes=all_queried_indexes,
                                    true_queried_indexes_baseline=all_queried_indexes_baseline)
        write_sequential_results_to_csv(results, opts)
    else:
        logger.debug("baseline:\n%s\norig iforest:\n%s" % (all_baseline, all_orig_iforest))

    if all_orig_num_seen is not None:
        prefix = opts.get_alad_metrics_name_prefix()
        orig_num_seen_file = os.path.join(opts.resultsdir, "%s-orig_num_seen.csv" % (prefix,))
        np.savetxt(orig_num_seen_file, all_orig_num_seen, fmt='%d', delimiter=',')

    if len(baseline_query_info) > 0:
        write_baseline_query_indexes(baseline_query_info, opts)

if run_tests:
    rng = np.random.RandomState(args.randseed)

    if mdl is None:
        # fit the model
        mdl = AadIsolationForest(n_estimators=opts.ifor_n_trees,
                                 max_samples=min(opts.ifor_n_samples, X_train.shape[0]),
                                 score_type=opts.ifor_score_type, random_state=rng,
                                 add_leaf_nodes_only=opts.ifor_add_leaf_nodes_only)
        mdl.fit(X_train)

    iforest_unit_tests_battery(X_train, labels, mdl, metrics, opts,
                               args.resultsdir, dataset_name=args.dataset)

