from alad_support import *


def write_sequential_results_to_csv(results, opts):
    """

    :param results: SequentialResults
    :param opts:
    :return:
    """
    prefix = opts.get_alad_metrics_name_prefix()
    num_seen_file = os.path.join(opts.resultsdir, "%s-num_seen.csv" % (prefix,))
    baseline_file = os.path.join(opts.resultsdir, "%s-baseline.csv" % (prefix,))
    queried_idxs_file = os.path.join(opts.resultsdir, "%s-queried.csv" % (prefix,))
    queried_idxs_baseline_file = os.path.join(opts.resultsdir, "%s-queried-baseline.csv" % (prefix,))
    aucs_file = os.path.join(opts.resultsdir, "%s-aucs.csv" % (prefix,))
    if results.num_seen is not None:
        np.savetxt(num_seen_file, results.num_seen, fmt='%d', delimiter=',')
    if results.num_seen_baseline is not None:
        np.savetxt(baseline_file, results.num_seen_baseline, fmt='%d', delimiter=',')
    if results.true_queried_indexes is not None:
        np.savetxt(queried_idxs_file, results.true_queried_indexes, fmt='%d', delimiter=',')
    if results.true_queried_indexes_baseline is not None:
        np.savetxt(queried_idxs_baseline_file, results.true_queried_indexes_baseline, fmt='%d', delimiter=',')
    if results.aucs is not None:
        np.savetxt(aucs_file, results.aucs, fmt='%f', delimiter=',')


def summarize_alad_to_csv(samples=None, ensembles=None, metrics=None, opts=None):

    allsamples = samples

    if allsamples is None:
        allsamples = load_all_samples(opts.dataset, opts.filedir, opts.get_fids(), opts)

    allensembles = ensembles
    if allensembles is None:
        allensembles = get_loda_alad_ensembles(opts.get_fids(), opts.get_runidxs(), allsamples, opts)

    allmetrics = metrics
    if allmetrics is None:
        allmetrics = consolidate_alad_metrics(opts.get_fids(), opts.get_runidxs(), opts)

    logger.debug(allmetrics.fids)
    logger.debug(allmetrics.runidxs)
    logger.debug(len(allensembles))
    logger.debug(len(allmetrics.metrics))

    # source(file.path(srcfolder,'loda/R','alad_routines.R'))
    alad_summary = summarize_alad_metrics(allensembles, allmetrics)
    opts.fid = 0
    opts.runidx = 0
    save_alad_summary(alad_summary, opts)
    write_sequential_results_to_csv(alad_summary, opts)
    # print("Completed alad-summary for %s" % (args.dataset,))


