from results_support import *
from test_setups import *


"""
python pyalad/test_results_support.py
"""


def summarize_alad_results():
    args = get_command_args(debug=False, debug_args=None)

    configure_logger(args)

    opts = Opts(args)

    logger.debug(opts.str_opts())

    set_seed(args.randseed)

    summarize_alad_to_csv(opts=opts)

    print "completed result summary %s for %s" % (opts.detector_type_str(), opts.dataset,)


def test_summarize_alad_results():
    datasets = get_test_datasets()

    for dataset in datasets:
        # args = get_command_args(debug=True, debug_args=test_args_alad_loda(dataset))
        args = get_command_args(debug=False, debug_args=None)

        configure_logger(args)

        opts = Opts(args)

        logger.debug(opts.str_opts())

        set_seed(args.randseed)

        # setup_subdirs(opts)

        summarize_alad_to_csv(opts=opts)

        print "completed result summary %s for %s" % (opts.detector_type_str(), opts.dataset,)

if __name__ == '__main__':
    summarize_alad_results()
