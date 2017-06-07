from r_support import *
from loda_support import *
from test_setups import *
from alad import *
from feature_bagging import *


"""
python pyalad/test_loda_units.py
"""


def test_load_model_csv(opts):
    opts.set_multi_run_options(1, 1)
    modelmanager = ModelManager.get_model_manager("csv")
    model = modelmanager.load_model(opts)
    print "model loaded..."


def test_load_data(args):
    filepath = os.path.join(args.filedir, args.dataset + "_1.csv")
    print filepath
    data = read_csv(filepath, header=True)
    print "data loaded..."


def test_load_samples(opts):
    alldata = load_all_samples(opts.dataset, opts.filedir, [1], opts)
    #logger.debug(alldata[0].lbls)
    #logger.debug(alldata[0].fmat)
    print "Loaded samples..."


def test_loda(opts):
    alldata = load_all_samples(opts.dataset, opts.filedir, [1], opts)
    #logger.debug(alldata[0].lbls)
    #logger.debug(alldata[0].fmat)
    print "Loaded samples..."

    a = alldata[0].fmat
    logger.debug(a.shape)

    if args.randseed > 0:
        np.random.seed(args.randseed)

    #args.original_dims = True
    lodares = loda(a, sparsity=opts.sparsity, mink=opts.mink, maxk=opts.maxk,
                   keep=opts.keep, exclude=opts.exclude, original_dims=opts.original_dims)

    print "Completed LODA..."


def pdf_hist_bin(x, h, minpdf=1e-8):
    """Returns the histogram bins for input values.

    Used for debugging only...
    """
    n = len(x)
    pd = np.zeros(n, dtype=int)
    for j in range(n):
        # use simple index lookup in case the histograms are equal width
        # this returns the lower index
        i = get_bin_for_equal_hist(h.breaks, x[j])
        if i >= len(h.density):
            i = len(h.density)-1  # maybe something else should be done here

        pd[j] = i+1  # adding 1 to make it like R  # max(h.density[i], minpdf)
    return pd


# get all bins from individual histograms.
def get_all_hist_pdf_bins(a, w, hists):
    """Returns the histogram bins for input values.

    Used for debugging only...
    """
    x = a.dot(w)
    bins = np.zeros(shape=(len(x), len(hists)), dtype=int)
    for i in range(len(hists)):
        bins[:, i] = pdf_hist_bin(x[:, i], hists[i])
    return bins


def test_show_baseline(opts):
    data = load_all_samples(opts.dataset, opts.filedir, range(opts.minfid, opts.maxfid + 1), opts)
    logger.debug("data loaded...")

    modelmanager = ModelManager.get_model_manager(opts.cachetype)
    ensembles = get_loda_alad_ensembles(range(opts.minfid, opts.maxfid + 1),
                                        range(1, opts.reruns + 1), data, opts)

    for i, fid in enumerate(range(opts.minfid, opts.maxfid+1)):
        fensembles = ensembles[i]
        for j, runidx in enumerate(range(1, opts.reruns+1)):

            opts.set_multi_run_options(fid, runidx)

            model = None
            if False:
                lodares = modelmanager.load_model(opts)
                model = generate_model_from_loda_result(lodares, data[i].fmat, data[i].lbls)
                ensemble = LodaEnsemble.ensemble_from_lodares(lodares, data[i].fmat, data[i].lbls)
                if False:
                    hpdfs = get_all_hist_pdfs(data[i].fmat, lodares.pvh.pvh.w, lodares.pvh.pvh.hists)
                    prefix = opts.model_file_prefix()
                    fname = os.path.join(opts.cachedir, "%s-hpdfs.csv" % (prefix,))
                    np.savetxt(fname, hpdfs, fmt='%.15f', delimiter=',')
                    logger.debug(lodares.pvh.pvh.hists[0].breaks)
                    logger.debug(lodares.pvh.pvh.hists[0].density)
            else:
                ensemble = fensembles[j]
            #logger.debug("model loaded...")

            #nll = np.mean(model.nlls, axis=1, dtype=float)
            #logger.debug("#scores: %d" % (len(nll),))
            #logger.debug(nll)
            #logger.debug("#anoms: %d" % (np.sum(data[0].lbls),))
            #logger.debug(data[0].lbls)

            if model is not None:
                ordered = order(model.anom_score, decreasing=True)
                logger.debug("\n%s" % (list(np.cumsum(model.lbls[ordered])[0:opts.budget]),))
            #logger.debug(data[0].lbls[ordered])
            ordered = order(ensemble.agg_scores, decreasing=True)
            logger.debug("\n%s" % (list(np.cumsum(ensemble.labels[ordered])[0:opts.budget]),))


def test_histogram(opts):
    data = load_all_samples(opts.dataset, opts.filedir, range(opts.minfid, opts.maxfid + 1), opts)
    logger.debug("data loaded...")

    logger.debug(ncol(data[0].fmat))
    for i in range(ncol(data[0].fmat)):
        x = data[0].fmat[:, i]
        #logger.debug(x)
        hist = histogram_r(x)
        #logger.debug(hist.breaks)
        #logger.debug(hist.counts)
        logger.debug(hist.density)


def test_ensemble_load(opts):
    samples = load_samples(opts.datafile, opts)
    logger.debug("Loaded samples...")
    em = PrecomputedEnsemble(opts)
    ensemble = em.load_data(samples.fmat, samples.lbls, opts)
    logger.debug("Loaded ensemble...")


def test_alad(opts):
    alad_results = alad(opts)
    write_sequential_results_to_csv(alad_results, opts)
    logger.debug("completed test_alad...")


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


def run_test(args):

    configure_logger(args)

    opts = Opts(args)

    logger.debug(opts.str_opts())

    if args.op == "load":
        test_load_data(opts)
    elif args.op == "samples":
        test_load_samples(opts)
    elif args.op == "loda":
        test_loda(opts)
    elif args.op == "csvmodel":
        test_load_model_csv(opts)
    elif args.op == "baseline":
        test_show_baseline(opts)
    elif args.op == "hist":
        test_histogram(opts)
    elif args.op == "ensemble":
        test_ensemble_load(opts)
    elif args.op == "alad":
        test_alad(opts)
    elif args.op == "aggregate_breadth_first":
        test_aggregate_scores_breadth_first(opts)
    else:
        raise ValueError("Invalid operation: %s" % (args.op,))
