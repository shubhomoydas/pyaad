from argparse import ArgumentParser
from r_support import *
from copy import copy

# ==============================
# Inference types
# ------------------------------
SIMPLE_UPD_TYPE = 1
SIMPLE_UPD_TYPE_R_OPTIM = 2
AAD_UPD_TYPE = 3
AAD_SLACK_CONSTR_UPD_TYPE = 4
BASELINE_UPD_TYPE = 5
AAD_ITERATIVE_GRAD_UPD_TYPE = 6

# Inference type names - first is blank string so these are 1-indexed
update_types = ["", "simple_online", "online_optim", "aad", "aad_slack", "baseline", "iter_grad"]
# ------------------------------

# ==============================
# Constraint types when Inference Type is AAD_PAIRWISE_CONSTR_UPD_TYPE
# ------------------------------
AAD_CONSTRAINT_PAIRWISE = 1  # slack vars [0, Inf]; weights [-Inf, Inf]
AAD_CONSTRAINT_PAIRWISE_WEIGHTS_POSITIVE_SUM_1 = 2  # slack vars [0, Inf]; weights [0, Inf]
AAD_CONSTRAINT_WEIGHTS_POSITIVE_SUM_1 = 3  # no pairwise; weights [0, Inf], sum(weights)=1

# Constraint type names - first is blank string so these are 1-indexed
constraint_types = ["", "pairwise", "pairwise_pos_wts_sum1", "pos_wts_sum1"]

# ==============================
# Baseline to use for simple weight inference
# ------------------------------
RELATIVE_MEAN = 1
RELATIVE_QUANTILE = 2

# first is blank to make the name list 1-indexed
RELATIVE_TO_NAMES = ["", "mean", "quantile"]
# ------------------------------


# ==============================
# Query types
# ------------------------------
QUERY_DETERMINISIC = 1
QUERY_BETA_ACTIVE = 2
QUERY_QUANTILE = 3
QUERY_RANDOM = 4
QUERY_SEQUENTIAL = 5

# first blank string makes the other names 1-indexed
query_type_names = ["", "top", "beta_active", "quantile", "random", "sequential"]
# ------------------------------


# ==============================
# Optimization libraries
# ------------------------------
OPTIMLIB_SCIPY = 'scipy'
OPTIMLIB_CVXOPT = 'cvxopt'
# ------------------------------


def get_option_list():
    parser = ArgumentParser()
    parser.add_argument("--filedir", action="store", default="",
                        help="Folder for input files")
    parser.add_argument("--cachedir", action="store", default="",
                        help="Folder where the generated models will be cached for efficiency")
    parser.add_argument("--plotsdir", action="store", default="",
                        help="Folder for output plots")
    parser.add_argument("--resultsdir", action="store", default="",
                        help="Folder where the generated metrics will be stored")
    parser.add_argument("--header", action="store_true", default=False,
                        help="Whether input file has header row")
    parser.add_argument("--startcol", action="store", type=int, default=2,
                        help="Starting column (1-indexed) for data in input CSV")
    parser.add_argument("--labelindex", action="store", type=int, default=1,
                        help="Index of the label column (1-indexed) in the input CSV. Lables should be anomaly/nominal")
    parser.add_argument("--dataset", action="store", default="", required=False,
                        help="Which dataset to use")
    parser.add_argument("--maxk", action="store", type=int, default=200,
                        help="Maximum number of random projections for LODA")
    parser.add_argument("--original_dims", action="store_true", default=False,
                        help="Whether to use original feature space instead of random projections")
    parser.add_argument("--randseed", action="store", type=int, default=42,
                        help="Random seed so that results can be replicated")
    parser.add_argument("--querytype", action="store", type=int, default=QUERY_DETERMINISIC,
                        help="Query strategy to use. 1 - Top, 2 - Beta-active, 3 - Quantile, 4 - Random")
    parser.add_argument("--reps", action="store", type=int, default=0,
                        help="Number of independent dataset samples to use")
    parser.add_argument("--reruns", action="store", type=int, default=0,
                        help="Number of times each sample dataset should be rerun with randomization")
    parser.add_argument("--runtype", action="store", type=str, default="simple",
                        help="[simple|multi] Whether the there are multiple sub-samples for the input dataset")
    parser.add_argument("--budget", action="store", type=int, default=35,
                        help="Number of feedback iterations")
    parser.add_argument("--maxbudget", action="store", type=int, default=100,
                        help="Maximum number of feedback iterations")
    parser.add_argument("--topK", action="store", type=int, default=0,
                        help="Top rank within which anomalies should be present")
    parser.add_argument("--tau", action="store", type=float, default=0.03,
                        help="Top quantile within which anomalies should be present. "
                             "Relevant only when topK<=0")
    parser.add_argument("--tau_nominal", action="store", type=float, default=0.5,
                        help="Top quantile below which nominals should be present. "
                             "Relevant only when simple quantile inference is used")
    parser.add_argument("--withprior", action="store_true", default=False,
                        help="Whether to use weight priors")
    parser.add_argument("--unifprior", action="store_true", default=False,
                        help="Whether to use uniform priors for weights. "
                             "By default, weight from previous iteration "
                             "is used as prior when --withprior is specified.")
    parser.add_argument("--batch", action="store_true", default=False,
                        help="Whether to query by active learning or select top ranked based on uniform weights")
    parser.add_argument("--sigma2", action="store", type=float, default=0.5,
                        help="If prior is used on weights, then the variance of prior")
    parser.add_argument("--Ca", action="store", type=float, default=100.,
                        help="Penalty for anomaly")
    parser.add_argument("--Cn", action="store", type=float, default=1.,
                        help="Penalty on nominals")
    parser.add_argument("--Cx", action="store", type=int, default=1000.,
                        help="Penalty on constraints")
    parser.add_argument("--inferencetype", action="store", type=int, default=AAD_UPD_TYPE,
                        help="Inference algorithm (simple_online(1) / online_optim(2) / aad_pairwise(3))")
    parser.add_argument("--constrainttype", action="store", type=int, default=AAD_CONSTRAINT_PAIRWISE,
                        help="Inference algorithm (simple_online(1) / online_optim(2) / aad_pairwise(3))")
    parser.add_argument("--pseudoanomrank_always", action="store_true", default=False,
                        help="Whether to always use pseudo anomaly instance")
    parser.add_argument("--max_anomalies_in_constraint_set", type=int, default=1000, required=False,
                        help="Maximum number of labeled anomaly instances to use in building pair-wise constraints")
    parser.add_argument("--max_nominals_in_constraint_set", type=int, default=1000, required=False,
                        help="Maximum number of labeled nominal instances to use in building pair-wise constraints")
    parser.add_argument("--relativeto", action="store", type=int, default=RELATIVE_MEAN,
                        help="The relative baseline for simple online (1=mean, 2=quantile)")
    parser.add_argument("--query_search_candidates", action="store", type=int, default=1,
                        help="Number of search candidates to use in each search state (when query_type=5)")
    parser.add_argument("--query_search_depth", action="store", type=int, default=1,
                        help="Depth of search tree (when query_type=5)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to enable output of debug statements")
    parser.add_argument("--log_file", type=str, default="", required=False,
                        help="File path to debug logs")
    parser.add_argument("--optimlib", type=str, default=OPTIMLIB_SCIPY, required=False,
                        help="optimization library to use")
    parser.add_argument("--op", type=str, default="nop", required=False,
                        help="name of operation")
    parser.add_argument("--cachetype", type=str, default="pydata", required=False,
                        help="type of cache (csv|pydata)")

    parser.add_argument("--scoresdir", type=str, default="", required=False,
                        help="Folder where precomputed scores from ensemble of detectors are stored in CSV format. "
                        "Applies only when runtype=simple")

    parser.add_argument("--ensembletype", type=str, default="regular", required=False,
                        help="[regular|loda] - 'regular' if the file has precomputed scores from ensembles; "
                             "'loda' if LODA projections are to be used as ensemble members. Note: LODA is stochastic, "
                             "hence multiple runs might be required to get an average estimate of accuracy.")
    parser.add_argument("--datafile", type=str, default="", required=False,
                        help="Original data in CSV format. This is used when runtype is 'regular'")
    parser.add_argument("--scoresfile", type=str, default="", required=False,
                        help="Precomputed scores from ensemble of detectors in CSV format. One detector per column;"
                             "first column has label [anomaly|nominal]")
    return parser


def get_command_args(debug=False, debug_args=None):
    parser = get_option_list()

    if debug:
        unparsed_args = debug_args
    else:
        unparsed_args = sys.argv
        if len(unparsed_args) > 0:
            unparsed_args = unparsed_args[1:len(unparsed_args)]  # script name is first arg

    args = parser.parse_args(unparsed_args)

    if args.startcol < 1:
        raise ValueError("startcol is 1-indexed and must be greater than 0")
    if args.labelindex < 1:
        raise ValueError("labelindex is 1-indexed and must be greater than 0")

    # LODA arguments
    args.keep = None
    args.exclude = None
    args.sparsity = np.nan
    args.explain = False
    #args.ntop = 30 # only for explanations
    args.marked = []

    return args


class Opts(object):
    def __init__(self, args):
        self.use_rel = False
        self.minfid = min(1, args.reps)
        self.maxfid = args.reps
        self.reruns = args.reruns
        self.runtype = args.runtype
        self.budget = args.budget
        self.maxbudget = args.maxbudget
        self.original_dims = args.original_dims
        self.qtype = args.querytype
        self.thres = 0.0  # used for feature weight in projection vector
        self.gam = 0.0  # used for correlation between projections
        self.nu = 1.0
        self.Ca = args.Ca  # 100.0,
        self.Cn = args.Cn
        self.Cx = args.Cx  # penalization for slack in pairwise constraints
        self.topK = args.topK
        self.tau = args.tau
        self.update_type = args.inferencetype
        self.constrainttype = args.constrainttype
        self.withprior = args.withprior  # whether to include prior in loss
        self.unifprior = args.unifprior
        self.priorsigma2 = args.sigma2  # 0.2, #0.5, #0.1,
        self.single_inst_feedback = False
        self.batch = args.batch
        self.pseudoanomrank_always = args.pseudoanomrank_always
        self.max_anomalies_in_constraint_set = args.max_anomalies_in_constraint_set
        self.max_nominals_in_constraint_set = args.max_nominals_in_constraint_set
        self.precision_k = [10, 20, 30]
        self.plot_hist = False
        self.relativeto = args.relativeto
        self.tau_nominal = args.tau_nominal
        self.query_search_candidates = args.query_search_candidates
        self.query_search_depth = args.query_search_depth
        self.optimlib = args.optimlib
        self.exclude = None
        self.keep = args.keep

        self.randseed = args.randseed

        # LODA specific
        self.mink = 100
        self.maxk = max(self.mink, args.maxk)
        self.sparsity = args.sparsity

        # file related options
        self.dataset = args.dataset
        self.header = args.header
        self.startcol = args.startcol
        self.filedir = args.filedir
        self.cachedir = args.cachedir
        self.resultsdir = args.resultsdir
        self.cachetype = args.cachetype
        self.fid = -1
        self.runidx = -1

        self.ensembletype = args.ensembletype
        self.datafile = args.datafile
        self.scoresdir = args.scoresdir
        self.scoresfile = args.scoresfile

    def is_simple_run(self):
        return self.runtype == "simple"

    def get_fids(self):
        if self.is_simple_run():
            return [0]
        else:
            return range(self.minfid, self.maxfid + 1)

    def get_runidxs(self):
        if self.is_simple_run():
            return [0]
        else:
            return range(1, self.reruns + 1)

    def set_multi_run_options(self, fid, runidx):
        self.fid = fid
        self.runidx = runidx

    def query_name_str(self):
        s = query_type_names[self.qtype]
        if self.qtype == QUERY_SEQUENTIAL:
            s = "%s_nc%d_d%d" % (s, self.query_search_candidates, self.query_search_depth)
        return s

    def update_type_str(self):
        s = update_types[self.update_type]
        if self.update_type == AAD_UPD_TYPE:
            return "%s_%s" % (s, constraint_types[self.constrainttype])
        else:
            return s

    def model_file_prefix(self):
        return "%s_%d_r%d" % (self.dataset, self.fid, self.runidx)

    def get_metrics_path(self):
        prefix = self.get_alad_metrics_name_prefix()
        return os.path.join(self.resultsdir, prefix + "_alad_metrics.pydata")

    def get_metrics_summary_path(self):
        prefix = self.get_alad_metrics_name_prefix()
        return os.path.join(self.resultsdir, prefix + "_alad_summary.pydata")

    def get_alad_metrics_name_prefix(self):
        if not self.is_simple_run():
            filesig = ("-fid%d" % (self.fid,)) + ("-runidx%d" % (self.runidx,))
        else:
            filesig = ""
        optimsig = "-optim_%s" % (self.optimlib,)
        nameprefix = (self.dataset +
                      ("-" + self.update_type_str()) +
                      ("_" + RELATIVE_TO_NAMES[self.relativeto] if self.update_type == SIMPLE_UPD_TYPE else "") +
                      ("-single" if self.single_inst_feedback else "") +
                      ("-" + self.query_name_str()) +
                      ("-orig" if self.original_dims else "") +
                      ("-batch" if self.batch else "-active") +
                      (("-unifprior" if self.unifprior else "-prevprior" + str(
                          self.priorsigma2)) if self.withprior else "-noprior") +
                      # ("-with_meanrel" if opts.withmeanrelativeloss else "-no_meanrel") +
                      ("-Ca%.0f" % (self.Ca,)) +
                      (("-Cn%0.0f" % (self.Cn,)) if self.Cn != 1 else "") +
                      ("-%d_%d" % (self.minfid, self.maxfid)) +
                      filesig +
                      ("-bd%d" % (self.budget,)) +
                      ("-tau%0.3f" % (self.tau,)) +
                      ("-tau_nominal" if self.update_type == SIMPLE_UPD_TYPE
                                         and self.relativeto == RELATIVE_QUANTILE
                                         and self.tau_nominal != 0.5 else "") +
                      ("-topK%d" % (self.topK,)) +
                      ("-pseudoanom_always_%s" % (self.pseudoanomrank_always,)) +
                      optimsig
                      )
        return nameprefix.replace(".", "_")

    def cached_loda_projections_path(self):
        """pre-computed cached projections path"""
        return os.path.join(self.cachedir, 'loda_projs')

    def str_opts(self):
        srr = (("[" + self.dataset + "]") +
               ("-%s" % (self.update_type_str(),)) +
               (("_%s" % (RELATIVE_TO_NAMES[self.relativeto],)) if self.update_type == SIMPLE_UPD_TYPE else "") +
               ("-single" if self.single_inst_feedback else "") +
               ("-query_" + self.query_name_str()) +
               ("-orig" if self.original_dims else "") +
               ("-batch" if self.batch else "-active") +
               ((("-unifprior" if self.unifprior else "-prevprior") + str(self.priorsigma2))
                if self.withprior else "-noprior") +
               ("-Ca" + str(self.Ca)) +
               (("-Cn" + str(self.Cn)) if self.Cn != 1 else "") +
               (("-Cx" + str(self.Cx)) if self.Cx != 1 else "") +
               ("-" + str(self.minfid) + "_" + str(self.maxfid)) +
               ("-reruns" + str(self.reruns)) +
               ("-bd" + str(self.budget)) +
               ("-tau" + str(self.tau)) +
               ("-tau_nominal" if self.update_type == SIMPLE_UPD_TYPE
                                  and self.relativeto == RELATIVE_QUANTILE
                                  and self.tau_nominal != 0.5 else "") +
               ("-topK" + str(self.topK)) +
               ("-pseudoanom_always_" + str(self.pseudoanomrank_always)) +
               ("-orgdim" if self.original_dims else "") +
               ("sngl_fbk" if self.single_inst_feedback else "") +
               ("-optimlib_%s" % (self.optimlib,))
               )
        return srr


def get_first_val_not_marked(vals, marked, start=1):
    for i in range(start, len(vals)):
        f = vals[i]
        if len(np.where(marked == f)[0]) == 0:
            return f
    return None


def get_first_vals_not_marked(vals, marked, n=1, start=1):
    unmarked = []
    for i in range(start, len(vals)):
        f = vals[i]
        if len(np.where(marked == f)[0]) == 0:
            unmarked.append(f)
        if len(unmarked) >= n:
            break
    return unmarked


def get_anomalies_at_top(scores, lbls, K):
    ordered_idxs = order(scores)
    sorted_lbls = lbls[ordered_idxs]
    counts = np.zeros(len(K))
    for i in range(len(K)):
        counts[i] = np.sum(sorted_lbls[1:K[i]])
    return counts


class SampleData(object):
    def __init__(self, lbls, fmat, fid):
        self.lbls = lbls
        self.fmat = fmat
        self.fid = fid


def load_samples(filepath, opts, fid=-1):
    """Loads the data file.

    :param filepath: str
    :param opts: Opts
    :param fid: int
    :return: SampleData
    """
    fdata = read_csv(filepath, header=opts.header)
    fmat = np.ndarray(shape=(fdata.shape[0], fdata.shape[1] - opts.startcol + 1), dtype=float)
    fmat[:, :] = fdata.iloc[:, (opts.startcol - 1):fdata.shape[1]]
    lbls = np.array([1 if v == "anomaly" else 0 for v in fdata.iloc[:, 0]], dtype=int)
    return SampleData(lbls=lbls, fmat=fmat, fid=fid)


def load_all_samples(dataset, dirpath, fids, opts):
    """
    Args:
        dataset:
        dirpath:
        fids:
        opts:
            opts.startcol: 1-indexed column number
            opts.labelindex: 1-indexed column number
    Returns: list
    """
    alldata = []
    for fid in fids:
        filename = "%s_%d.csv" % (dataset, fid)
        filepath = os.path.join(dirpath, filename)
        alldata.append(load_samples(filepath, opts, fid=fid))
    return alldata

