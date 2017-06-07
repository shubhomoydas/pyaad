from unit_tests import *


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
        # "--inferencetype=%d" % (SIMPLE_UPD_TYPE,),
        # "--inferencetype=%d" % (SIMPLE_UPD_TYPE_R_OPTIM,),
        "--inferencetype=%d" % (AAD_UPD_TYPE,),
        # "--inferencetype=%d" % (AAD_SLACK_CONSTR_UPD_TYPE,),
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
        , "--log_file=temp/pyalad.log"
        , "--filedir=datasets/anomaly/%s/fullsamples" % (dataset,)
        , "--cachedir=datasets/anomaly/%s/fullmodel/pyalad" % (dataset,)
        , "--resultsdir=datasets/anomaly/%s/fullresults/pyalad/aad_pairwise-featurebag" % (dataset,)
        , "--plotsdir=datasets/anomaly/%s/fullplots" % (dataset,)
        # , "--filedir=datasets/anomaly/%s/fullsamples" % (dataset,)
        # , "--cachedir=datasets/anomaly/%s/fullmodel" % (dataset,)
        # , "--resultsdir=datasets/anomaly/%s/fullresults" % (dataset,)
        # , "--plotsdir=datasets/anomaly/%s/fullplots" % (dataset,)
        #
        # , "--ensembletype=loda"
        , "--ensembletype=regular"
        #
        , "--runtype=multi"
        # , "--runtype=simple"
        , "--datafile=datasets/anomaly/%s/featurebag_lof/%s_1.csv" % (dataset, dataset)
        , "--scoresfile=datasets/anomaly/%s/featurebag_lof/%s_1.csv" % (dataset, dataset)
        #
        , "--scoresdir=datasets/anomaly/%s/featurebag_lof" % (dataset,)
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
        run_test(args)
        print "Completed %s" % (dataset, )


