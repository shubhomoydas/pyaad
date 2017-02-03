from alad_support import *
from results_support import *
from test_setups import *


"""
python pyalad/alad.py
"""


if __name__ == '__main__':

    args = get_command_args(debug=False, debug_args=None)

    configure_logger(args)

    opts = Opts(args)

    logger.debug(opts.str_opts())
    # print opts.str_opts()

    set_seed(opts.randseed)

    samples, ensembles, metrics = alad(opts)

    summarize_alad_to_csv(samples=samples, ensembles=ensembles, metrics=metrics, opts=opts)

    print "completed alad %s for %s" % (opts.update_type_str(), opts.dataset,)

