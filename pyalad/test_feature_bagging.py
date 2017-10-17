"""
This code merely aggregates precomputed scores from an ensemble.

Related publication:
    Feature Bagging for Outlier Detection by Aleksandar Lazarevic and Vipin Kumar, KDD 2005.
"""

from alad_support import *
from unit_tests import *


def aggregate_scores_breadth_first(opts):

    scores = load_ensemble_scores(opts.scoresfile, header=True)

    n, m = scores.fmat.shape

    score_ranks = np.zeros(shape=(n, m), dtype=int)
    for i in range(m):
        score_ranks[:, i] = order(scores.fmat[:, i], decreasing=True)

    # logger.debug(list(scores.fmat[score_ranks[:, 0], 0]))

    bt = get_budget_topK(n, opts)
    final_ranks = np.zeros(nrow(scores.fmat), dtype=int) - 1
    fp = 0
    for i in range(n):
        insts_at_rank_i = score_ranks[i, :]
        for j in insts_at_rank_i:
            pos = np.where(final_ranks == j)[0]
            if len(pos) == 0:
                final_ranks[fp] = j
                fp += 1
        if fp >= n:
            break

    queried = list(final_ranks[0:bt.budget])
    # the first two positions are (fid,runidx) = (0,0)
    num_seen = np.zeros(len(queried) + 2)
    num_seen[2:len(num_seen)] = np.cumsum(scores.lbls[queried])
    num_seen = num_seen.reshape((1, len(num_seen)))
    logger.debug(num_seen.shape)

    return num_seen


def test_aggregate_scores_breadth_first(opts):
    if opts.is_simple_run():
        num_seen = aggregate_scores_breadth_first(opts)
        prefix = opts.get_alad_metrics_name_prefix()
        num_seen_file = os.path.join(opts.resultsdir, "%s-breadth_first.csv" % (prefix,))
        np.savetxt(num_seen_file, num_seen, fmt='%d', delimiter=',')
    else:
        fids = opts.get_fids()
        runidxs = opts.get_runidxs()
        n = len(fids) * len(runidxs)
        num_seen = np.zeros(shape=(n, opts.budget+2))
        i = 0
        for fid in fids:
            for runidx in runidxs:
                opts.set_multi_run_options(fid, runidx)
                opts.scoresfile = os.path.join(opts.scoresdir,
                                               "%s_%d_%d.csv" % (opts.dataset, fid, runidx))
                num_seen[i, :] = aggregate_scores_breadth_first(opts)
                i += 1

        opts.set_multi_run_options(0, 0)
        prefix = opts.get_alad_metrics_name_prefix()
        num_seen_file = os.path.join(opts.resultsdir, "%s-breadth_first.csv" % (prefix,))
        np.savetxt(num_seen_file, num_seen, fmt='%d', delimiter=',')


def test_args_feature_bagging(dataset, op="nop"):
    test_args = [
        "--startcol=2", "--labelindex=1", "--header", "--randseed=42",
        "--dataset=" + dataset,
        #
        "--querytype=%d" % (QUERY_DETERMINISIC,),
        # "--querytype=%d" % (QUERY_BETA_ACTIVE,),
        # "--querytype=%d" % (QUERY_QUANTILE,),
        # "--querytype=%d" % (QUERY_RANDOM,),
        # "--querytype=%d" % (QUERY_SEQUENTIAL,),
        #
        # "--detector_type=%d" % (SIMPLE_UPD_TYPE,),
        # "--detector_type=%d" % (SIMPLE_UPD_TYPE_R_OPTIM,),
        "--detector_type=%d" % (AAD_UPD_TYPE,),
        # "--detector_type=%d" % (AAD_SLACK_CONSTR_UPD_TYPE,),
        #
        "--constrainttype=%d" % (AAD_CONSTRAINT_PAIRWISE,),
        # "--constrainttype=%d" % (AAD_CONSTRAINT_PAIRWISE_WEIGHTS_POSITIVE_SUM_1,),
        # "--constrainttype=%d" % (AAD_CONSTRAINT_WEIGHTS_POSITIVE_SUM_1,),
        #
        "--sigma2=0.5",
        "--reps=1"
        , "--reruns=10"
        , "--budget=60"
        , "--maxbudget=100"
        , "--topK=0"
        , "--relativeto=1"
        , "--tau=0.03"
        , "--tau_nominal=0.03"
        , "--withprior"
        , "--unifprior"
        , "--Ca=100"
        , "--Cn=1"
        , "--Cx=1000"
        , "--query_search_candidates=3"
        , "--query_search_depth=3"
        , "--debug"
        , "--op=%s" % (op,)
        #
        # , "--cachetype=csv"
        , "--cachetype=pydata"
        #
        # , "--optimlib=scipy"
        , "--optimlib=cvxopt"
        #
        # , "--withmeanrelativeloss"
        , "--log_file=/Users/moy/work/temp/pyalad.log"
        , "--filedir=/Users/moy/work/datasets/anomaly/%s/fullsamples" % (dataset,)
        , "--cachedir=/Users/moy/work/datasets/anomaly/%s/fullmodel/pyalad" % (dataset,)
        , "--resultsdir=/Users/moy/work/datasets/anomaly/%s/fullresults/pyalad/aad_pairwise-featurebag" % (dataset,)
        , "--plotsdir=/Users/moy/work/datasets/anomaly/%s/fullplots" % (dataset,)
        # , "--filedir=/scratch/datasets/anomaly/%s/fullsamples" % (dataset,)
        # , "--cachedir=/scratch/datasets/anomaly/%s/fullmodel" % (dataset,)
        # , "--resultsdir=/scratch/datasets/anomaly/%s/fullresults" % (dataset,)
        # , "--plotsdir=/scratch/datasets/anomaly/%s/fullplots" % (dataset,)
        #
        # , "--ensembletype=loda"
        , "--ensembletype=regular"
        #
        , "--runtype=multi"
        # , "--runtype=simple"
        , "--datafile=/Users/moy/work/datasets/anomaly/%s/featurebag_lof/%s_1.csv" % (dataset, dataset)
        , "--scoresfile=/Users/moy/work/datasets/anomaly/%s/featurebag_lof/%s_1.csv" % (dataset, dataset)
        #
        , "--scoresdir=/Users/moy/work/datasets/anomaly/%s/featurebag_lof" % (dataset,)
    ]
    return test_args


if __name__ == '__main__':

    # op = "alad"
    op = "aggregate_breadth_first"
    datasets = [
        "abalone",
        "ann_thyroid_1v3",
        "cardiotocography_1",
        "covtype_sub",
        "kddcup_sub",
        "mammography_sub",
        "shuttle_sub", "yeast"
    ]
    # datasets = ["abalone"]

    for dataset in datasets:
        args = get_command_args(debug=True, debug_args=test_args_feature_bagging(dataset, op))
        if not os.path.isdir(args.resultsdir):
            logger.debug("creating folder: %s" % (args.resultsdir,))
            dir_create(args.resultsdir)
        test_aggregate_scores_breadth_first(args)
        print "Completed %s" % (dataset, )


