import os
import numpy as np

import logging

from app_globals import *
from alad_support import *
from r_support import matrix, cbind

from forest_aad_detector import *
from forest_aad_support import prepare_forest_aad_debug_args
from results_support import write_sequential_results_to_csv
from data_stream import *

"""
To debug:
    pythonw pyalad/forest_aad_stream.py
"""

logger = logging.getLogger(__name__)


class StreamingAnomalyDetector(object):
    """
    Attributes:
        model: AadForest
        stream: DataStream
        max_buffer: int
            Determines the window size
        buffer_instances_x: list
    """
    def __init__(self, stream, model, labeled_x=None, labeled_y=None,
                 unlabeled_x=None, unlabeled_y=None, max_buffer=512):
        self.model = model
        self.stream = stream
        self.max_buffer = max_buffer

        self.buffer_x = None
        self.buffer_y = None

        self.unlabeled_x = unlabeled_x
        self.unlabeled_y = unlabeled_y

        self.labeled_x = labeled_x
        self.labeled_y = labeled_y

        self.qstate = None

    def reset_buffer(self):
        self.buffer_x = None
        self.buffer_y = None

    def add_buffer_xy(self, x, y):
        if self.buffer_x is None:
            self.buffer_x = x
        else:
            self.buffer_x = rbind(self.buffer_x, x)

        if self.buffer_y is None:
            self.buffer_y = y
        else:
            if y is not None:
                self.buffer_y = append(self.buffer_y, y)

    def move_buffer_to_unlabeled(self):
        self.unlabeled_x = self.buffer_x
        self.unlabeled_y = self.buffer_y
        self.reset_buffer()

    def get_num_instances(self):
        """Returns the total number of labeled and unlabeled instances that will be used for weight inference"""
        n = 0
        if self.unlabeled_x is not None:
            n += nrow(self.unlabeled_x)
        if self.labeled_x is not None:
            logger.debug("labeled_x: %s" % str(self.labeled_x.shape))
            n += nrow(self.labeled_x)
        return n

    def init_query_state(self, opts):
        n = self.get_num_instances()
        bt = get_budget_topK(n, opts)
        self.qstate = Query.get_initial_query_state(opts.qtype, opts=opts, qrank=bt.topK,
                                                    a=1., b=1., budget=bt.budget)

    def get_next_from_stream(self, n=0):
        if n == 0:
            n = self.max_buffer
        x, y = self.stream.read_next_from_stream(n)

        if False:
            if self.buffer_x is not None:
                logger.debug("buffer shape: %s" % str(self.buffer_x.shape))
            logger.debug("x.shape: %s" % str(x.shape))

        self.add_buffer_xy(x, y)

        self.model.add_samples(x, current=False)

        return x, y

    def update_model_from_buffer(self):
        self.model.update_model_from_stream_buffer()

    def get_next_transformed(self, n=1):
        x, y = self.get_next_from_stream(n)
        x_new = self.model.transform_to_region_features(x, dense=False)
        return x_new, y

    def stream_buffer_empty(self):
        return self.stream.empty()

    def get_anomaly_scores(self, x):
        x_new = self.model.transform_to_region_features(x, dense=False)
        scores = self.model.get_score(x_new)
        return scores

    def setup_data_for_feedback(self):
        """
        Prepares the input matrices/data structures for weight update. The format
        is such that the top rows of data matrix are labeled and below are unlabeled.

        :return: (np.ndarray, np.array, np.array, np.array)
            (x, y, ha, hn)
            x - data matrix, y - labels (np.nan for unlabeled),
            ha - indexes of labeled anomalies, hn - indexes of labeled nominals
        """
        x = None
        y = None
        if self.labeled_x is not None:
            x = self.labeled_x.copy()
            y = self.labeled_y.copy()
            ha = np.where(self.labeled_y == 1)[0]
            hn = np.where(self.labeled_y == 0)[0]
        else:
            ha = np.zeros(0, dtype=int)
            hn = np.zeros(0, dtype=int)
        if self.unlabeled_x is not None:
            if x is None:
                x = self.unlabeled_x.copy()
            else:
                x = np.append(x, self.unlabeled_x, axis=0)
            if self.unlabeled_y is not None:
                if y is not None:
                    y = np.append(y, self.unlabeled_y)
                else:
                    y = self.unlabeled_y.copy()
            else:
                if y is not None:
                    y = np.append(y, np.ones(nrow(self.unlabeled_x), dtype=int) * -1)
                else:
                    y = np.ones(nrow(self.unlabeled_x), dtype=int) * -1
        if True:
            logger.debug("x: %d, y: %d, ha: %d, hn:%d" % (nrow(x), len(y), len(ha), len(hn)))
        return x, y, ha, hn

    def get_num_labeled(self):
        """Returns the number of instances for which we already have label feedback"""
        if self.labeled_y is not None:
            return len(self.labeled_y)
        return 0

    def get_query_data(self, x=None, y=None, ha=None, hn=None):
        """Returns the best instance that should be queried, along with other data structures"""
        n = self.get_num_instances()
        n_feedback = self.get_num_labeled()
        if True:
            logger.debug("get_query_data() n: %d, n_feedback: %d" % (n, n_feedback))
        if n == 0:
            raise ValueError("No instances available")
        if x is None:
            x, y, ha, hn = self.setup_data_for_feedback()
        x_transformed = model.transform_to_region_features(x, dense=False)
        order_anom_idxs, anom_score = self.model.order_by_score(x_transformed)
        xi = self.qstate.get_next_query(maxpos=n, ordered_indexes=order_anom_idxs,
                                        queried_items=np.arange(n_feedback),
                                        x=x_transformed, lbls=y, anom_score=anom_score,
                                        w=self.model.w, hf=append(ha, hn),
                                        remaining_budget=opts.budget - n_feedback)
        if True:
            logger.debug("ordered instances[%d]: %s\nha: %s\nhn: %s\nxi: %d" %
                         (opts.budget, str(list(order_anom_idxs[0:opts.budget])),
                          str(list(ha)), str(list(hn)), xi))
        return xi, x, y, x_transformed, ha, hn

    def move_unlabeled_to_labeled(self, xi, yi):
        unlabeled_idx = xi - self.get_num_labeled()

        self.labeled_x = rbind(self.labeled_x, matrix(self.unlabeled_x[unlabeled_idx], nrow=1))
        if self.labeled_y is None:
            self.labeled_y = np.array([yi], dtype=int)
        else:
            self.labeled_y = np.append(self.labeled_y, [yi])
        mask = np.ones(self.unlabeled_x.shape[0], dtype=bool)
        mask[unlabeled_idx] = False
        self.unlabeled_x = self.unlabeled_x[mask]
        self.unlabeled_y = self.unlabeled_y[mask]

    def update_weights_with_feedback(self, xi, yi, x, y, x_transformed, ha, hn, opts):
        """Relearns the optimal weights from feedback and updates internal labeled and unlabeled matrices

        IMPORTANT:
            This API assumes that the input x, y, x_transformed are consistent with
            the internal labeled/unlabeled matrices, i.e., the top rows/values in
            these matrices are from labeled data and bottom ones are from internally
            stored unlabeled data.
        """

        # Add the newly labeled instance to the corresponding list of labeled
        # instances and remove it from the unlabeled set.
        sad.move_unlabeled_to_labeled(xi, yi)

        if yi == 1:
            ha = append(ha, [xi])
        else:
            hn = append(hn, [xi])

        self.model.update_weights(x_transformed, y, ha, hn, opts)


def get_rearranging_indexes(add_pos, move_pos, n):
    """Creates an array 0...n-1 and moves value at 'move_pos' to 'add_pos', and shifts others back

    Useful to reorder data when we want to move instances from unlabeled set to labeled.
    TODO:
        Use this to optimize the API StreamingAnomalyDetector.get_query_data()
        since it needs to repeatedly convert the data to transformed [node] features.

    Example:
        get_rearranging_indexes(2, 2, 10):
            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        get_rearranging_indexes(0, 1, 10):
            array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])

        get_rearranging_indexes(2, 9, 10):
            array([0, 1, 9, 2, 3, 4, 5, 6, 7, 8])

    :param add_pos:
    :param move_pos:
    :param n:
    :return:
    """
    if add_pos > move_pos:
        raise ValueError("add_pos must be less or equal to move_pos")
    rearr_idxs = np.arange(n)
    if add_pos == move_pos:
        return rearr_idxs
    rearr_idxs[(add_pos + 1):(move_pos + 1)] = rearr_idxs[add_pos:move_pos]
    rearr_idxs[add_pos] = move_pos
    return rearr_idxs


def read_data(opts):
    data = DataFrame.from_csv(opts.datafile, header=0, sep=',', index_col=None)
    X_train = np.zeros(shape=(data.shape[0], data.shape[1] - 1))
    for i in range(X_train.shape[1]):
        X_train[:, i] = data.iloc[:, i + 1]
    labels = np.array([1 if data.iloc[i, 0] == "anomaly" else 0 for i in range(data.shape[0])], dtype=int)
    return X_train, labels


def train_aad_model(opts, X_train):
    rng = np.random.RandomState(opts.randseed + opts.fid * opts.reruns + opts.runidx)
    # fit the model
    model = AadForest(n_estimators=opts.forest_n_trees,
                      max_samples=min(opts.forest_n_samples, X_train.shape[0]),
                      score_type=opts.forest_score_type, random_state=rng,
                      add_leaf_nodes_only=opts.forest_add_leaf_nodes_only,
                      max_depth=opts.forest_max_depth,
                      ensemble_score=opts.ensemble_score,
                      detector_type=opts.detector_type, n_jobs=opts.n_jobs)
    model.fit(X_train)
    return model


def prepare_aad_model(X, y, opts):
    if opts.load_model and opts.modelfile != "" and os.path.isfile(opts.modelfile):
        logger.debug("Loading model from file %s" % opts.modelfile)
        model = load_aad_model(opts.modelfile)
    else:
        model = train_aad_model(opts, X)

    logger.debug("total #nodes: %d" % (len(model.all_regions)))
    if False:
        if model.w is not None:
            logger.debug("w:\n%s" % str(list(model.w)))
        else:
            logger.debug("model weights are not set")
    return model


def run_feedback(streamAD, n_runs, opts):
    """

    :param streamAD: StreamingAnomalyDetector
    :param n_runs: int
    :param opts: Opts
    :return:
    """
    # get baseline metrics
    x_transformed = model.transform_to_region_features(streamAD.unlabeled_x, dense=False)
    ordered_idxs, _ = streamAD.model.order_by_score(x_transformed)
    seen_baseline = streamAD.unlabeled_y[ordered_idxs[0:n_runs]]
    num_seen_baseline = np.cumsum(seen_baseline)
    logger.debug("num_seen_baseline:\n%s" % str(list(num_seen_baseline)))

    for i in np.arange(n_runs):
        xi, x, y, x_transformed, ha, hn = sad.get_query_data()
        sad.update_weights_with_feedback(xi, y[xi], x, y, x_transformed, ha, hn, opts)
        logger.debug("\nha: %s\nhn: %s" % (str(list(ha)), str(list(hn))))
        logger.debug("y:\n%s" % str(list(y)))
    logger.debug("w:\n%s" % str(list(streamAD.model.w)))


if False:
    # DEBUG
    args = prepare_forest_aad_debug_args()
else:
    # PRODUCTION
    args = get_command_args(debug=False)
# print "log file: %s" % args.log_file
configure_logger(args)

opts = Opts(args)
# print opts.str_opts()
logger.debug(opts.str_opts())

if not opts.streaming:
    raise ValueError("Only streaming supported")

X_full, y_full = read_data(opts)
# X_train = X_train[0:10, :]
# labels = labels[0:10]

logger.debug("loaded file: (%s) %s" % (str(X_full.shape), opts.datafile))
logger.debug("results dir: %s" % opts.resultsdir)

all_num_seen_baseline = None
all_queried_baseline = None
aucs = np.zeros(0, dtype=float)

opts.fid = 1
for runidx in opts.get_runidxs():
    opts.set_multi_run_options(opts.fid, runidx)

    stream = DataStream(X_full, y_full)
    X_train, y_train = stream.read_next_from_stream(opts.stream_window)

    # logger.debug("X_train:\n%s\nlabels:\n%s" % (str(X_train), str(list(labels))))

    model = prepare_aad_model(X_train, y_train, opts)  # initial model training
    sad = StreamingAnomalyDetector(stream, model, unlabeled_x=X_train, unlabeled_y=y_train,
                                   max_buffer=opts.stream_window)
    sad.init_query_state(opts)

    if True:
        run_feedback(sad, opts.budget, opts)
        print "This is experimental/demo code for streaming integration and will be application specific." + \
              " Exiting after reading max %d instances from stream and iterating for %d feedback..." % \
                (opts.stream_window, opts.budget)
        exit(0)

    all_scores = np.zeros(0)
    all_y = np.zeros(0)

    scores = sad.get_anomaly_scores(X_train)
    # auc = fn_auc(cbind(y_train, -scores))
    all_scores = np.append(all_scores, scores)
    all_y = np.append(all_y, y_train)
    iter = 0
    while not sad.stream_buffer_empty():
        iter += 1

        xi, x, y, x_transformed, ha, hn = sad.get_query_data()

        sad.update_weights_with_feedback(xi, y[xi], x, y, x_transformed, ha, hn, opts)
        # logger.debug("updated weights:\n%s" % str(list(model.w)))

        x_eval, y_eval = sad.get_next_from_stream(sad.max_buffer)
        scores = sad.get_anomaly_scores(x_eval)  # compute scores before updating the model

        all_scores = np.append(all_scores, scores)
        all_y = np.append(all_y, y_eval)

        if opts.allow_stream_update:
            sad.update_model_from_buffer()

        sad.move_buffer_to_unlabeled()
        # logger.debug("iter %d" % iter)

    auc = fn_auc(cbind(all_y, -all_scores))
    logger.debug("AUC: %f" % auc)
    aucs = append(aucs, [auc])

    queried_baseline = order(all_scores, decreasing=True)[0:opts.budget]
    num_seen_baseline = np.cumsum(all_y[queried_baseline])
    logger.debug("Numseen in %d budget:\n%s" % (opts.budget, str(list(num_seen_baseline))))

    queried_baseline = append(np.array([opts.fid, opts.runidx], dtype=queried_baseline.dtype), queried_baseline)
    num_seen_baseline = append(np.array([opts.fid, opts.runidx], dtype=num_seen_baseline.dtype), num_seen_baseline)
    all_queried_baseline = rbind(all_queried_baseline, matrix(queried_baseline, nrow=1))
    all_num_seen_baseline = rbind(all_num_seen_baseline, matrix(num_seen_baseline, nrow=1))

    logger.debug("Completed runidx: %d" % runidx)

results = SequentialResults(num_seen_baseline=all_num_seen_baseline,
                            true_queried_indexes_baseline=all_queried_baseline, aucs=aucs)
write_sequential_results_to_csv(results, opts)
