from results_support import *


def setup_subdirs(opts):
    if not os.path.isdir(opts.cachedir):
        logger.debug("creating folder: %s" % (opts.cachedir,))
        dir_create(opts.cachedir)

    resdir = os.path.join(opts.resultsdir, opts.detector_type_str())
    if not os.path.isdir(resdir):
        logger.debug("creating folder: %s" % (resdir,))
        dir_create(resdir)
    opts.resultsdir = resdir

    return opts


def get_test_datasets():
    # datasets = ["toy"]
    # datasets = ["abalone"]
    # datasets = ["ann_thyroid_1v3"]
    # datasets = ["cardiotocography"]
    datasets = ["abalone", "ann_thyroid_1v3",
                "covtype_sub", # "kddcup_sub",
                "mammography_sub", "shuttle_sub",
                "yeast", "cardiotocography_1"]
    # datasets = ["kddcup_sub"]
    datasets = ["cardiotocography_1"]
    return datasets


def test_args_alad_loda(dataset, op="nop"):
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
        , "--optimlib=scipy"
        # , "--optimlib=cvxopt"
        #
        # , "--withmeanrelativeloss"
        , "--log_file=./temp/pyaad.log"
        , "--filedir=./datasets/anomaly/%s/fullsamples" % (dataset,)
        , "--cachedir=./temp/%s/fullmodel/pyalad" % (dataset,)
        , "--resultsdir=./temp/%s/fullresults/pyalad" % (dataset,)
        , "--plotsdir=./temp/%s/fullplots" % (dataset,)
        #
        , "--ensembletype=loda"
        # , "--ensembletype=regular"
        #
        , "--runtype=multi"
        # , "--runtype=simple"
        , "--datafile=./datasets/anomaly/%s/fullsamples/%s_1.csv" % (dataset, dataset)
        , "--scoresfile=./datasets/anomaly/%s/featurebag_lof/%s_1.csv" % (dataset, dataset)
    ]
    return test_args


