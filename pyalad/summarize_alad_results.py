from alad_support import *
from results_support import *
from test_setups import *


"""
python pyalad/summarize_alad_results.py
"""


def summarize_alad_results(samples=None, ensembles=None, metrics=None, opts=None):

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
    #save_alad_summary(alad_summary, opts)
    write_sequential_results_to_csv(alad_summary, opts)
    # print("Completed alad-summary for %s" % (args.dataset,))


if __name__ == '__main__':

    args = get_command_args(debug=False, debug_args=None)

    configure_logger(args)

    opts = Opts(args)

    logger.debug(opts.str_opts())
    # print opts.str_opts()

    set_seed(opts.randseed)

    summarize_alad_results(opts=opts)

    opts.fid = 0
    opts.runidx = 0
    #write_sequential_results_to_csv(alad_results, opts)

    print "completed alad %s for %s" % (opts.detector_type_str(), opts.dataset,)

