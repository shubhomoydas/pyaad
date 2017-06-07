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

    alad_results = alad(opts)

    opts.fid = 0
    opts.runidx = 0
    write_sequential_results_to_csv(alad_results, opts)

    print "completed alad %s for %s" % (opts.update_type_str(), opts.dataset,)

