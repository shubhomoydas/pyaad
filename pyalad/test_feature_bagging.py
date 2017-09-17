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
        run_test(args)
        print "Completed %s" % (dataset, )


