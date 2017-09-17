from alad_support import *
from results_support import *
from test_setups import *


"""
python pyalad/test_alad.py --op=all
"""


def alad_dataset(dataset):

    args = get_command_args(debug=True, debug_args=test_args_alad_loda(dataset))
    configure_logger(args)

    opts = Opts(args)

    logger.debug(opts.str_opts())

    set_seed(opts.randseed)

    setup_subdirs(opts)

    samples, ensembles, metrics = alad(opts)

    summarize_alad_to_csv(samples=samples, ensembles=ensembles, metrics=metrics, opts=opts)

    print "completed alad %s for %s" % (opts.detector_type_str(), opts.dataset,)


def alad_multiple_datasets():
    datasets = get_test_datasets()

    for dataset in datasets:
        alad_dataset(dataset)


def test_alad_single():
    args = get_command_args(debug=True, debug_args=[])

    opts = Opts(args)
    opts.dataset = "kddcup_sub"
    opts.budget = 60

    opts.withprior = True
    opts.unifprior = True

    opts.filedir = "/Users/moy/work/datasets/anomaly/%s/fullsamples" % (opts.dataset,)
    opts.cachedir = "/Users/moy/work/datasets/anomaly/%s/fullmodel/pyalad" % (opts.dataset,)
    # opts.resultsdir = "/Users/moy/work/datasets/anomaly/%s/fullresults/pyalad" % (opts.dataset,)
    opts.header = True
    opts.mink = 100

    # opts.datafile = "/Users/moy/work/datasets/anomaly/%s/fullsamples/%s_1" % (opts.dataset, opts.dataset)
    # opts.scoresfile = "/Users/moy/work/datasets/anomaly/%s/fullsamples" % (opts.dataset,)

    opts.ensembletype = "loda"
    opts.runtype = "multi"
    opts.debug = True

    logger.debug(opts.str_opts())

    fid = 1

    filename = "%s_%d.csv" % (opts.dataset, fid)
    filepath = os.path.join(opts.filedir, filename)
    s = load_samples(filepath, opts, fid)

    logger.debug("shape of samples: %s" % str(s.fmat.shape))

    for runidx in range(0, 10):
        opts.set_multi_run_options(fid, runidx)
        np.random.seed(42 + runidx)
        em = EnsembleManager.get_ensemble_manager(opts)

        ensemble = em.load_data(s.fmat, s.lbls, opts)

        if ensemble.model is not None:
            m = ncol(ensemble.model.w)
            d = nrow(ensemble.model.w)
            logger.debug("shape of projections: %s" % str(ensemble.model.w.shape))
            # logger.debug(ensemble.model.w[:, 0].reshape((d,)))
            # logger.debug(np.where(ensemble.model.w[:, 0].reshape((d,)) == 0))
            nonzeros = [d - len(np.where(ensemble.model.w[:, k].reshape((d,)) == 0)[0]) for k in range(m)]
            logger.debug("Non-zeros: \n%s" % str(nonzeros))
            #for k in range(m):
            #    # logger.debug(list(np.round(ensemble.model.w[:, k], 4)))
            #    logger.debug("Non-zeros: %d" % (m - len(np.argwhere(ensemble.model.w[:, k] == 0)[0]),))

        ordered = order(ensemble.agg_scores, decreasing=True)
        logger.debug("\n%s" % (list(np.cumsum(ensemble.labels[ordered])[0:opts.budget]),))


if __name__ == '__main__':

    args = get_command_args(debug=False, debug_args=test_args_alad_loda(get_test_datasets()[0]))
    configure_logger(args)

    if args.op == "all":
        alad_multiple_datasets()
    elif args.op == "one":
        alad_dataset(args.dataset)
    elif args.op == "prod":
        # run with options provided from commandline
        opts = Opts(args)
        alad_results = alad(opts)
        write_sequential_results_to_csv(alad_results, opts)
        print "completed alad %s for %s" % (opts.detector_type_str(), args.dataset,)
    elif args.op == "single":
        test_alad_single()
    else:
        raise ValueError("Invaid operation %s" % (args.op,))
