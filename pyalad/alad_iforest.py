from copy import deepcopy
import numpy as np
from scipy.sparse import lil_matrix
from scipy import sparse
from scipy.sparse import csr_matrix, vstack
from sklearn.ensemble import IsolationForest

import logging

from app_globals import *  # get_command_args, Opts, configure_logger
from alad_support import *
from r_support import matrix, cbind
import numbers
from alad_iforest_loss import *


class RegionData(object):
    def __init__(self, region, path_length, node_id, score, node_samples):
        self.region = region
        self.path_length = path_length
        self.node_id = node_id
        self.score = score
        self.node_samples = node_samples


def is_in_region(x, region):
    d = len(x)
    for i in range(d):
        if not region[i][0] <= x[i] <= region[i][1]:
            return False
    return True


def transform_features(x, all_regions, d):
    """ Inefficient method for looking up region membership.

    Note: This method is only for DEBUG. For a faster
    implementation, see below.
    @see: AadIsolationForest.transform_to_region_features

    :param x:
    :param all_regions:
    :param d:
    :return:
    """
    # translate x's to new coordinates
    x_new = np.zeros(shape=(x.shape[0], len(d)), dtype=np.float64)
    for i in range(x.shape[0]):
        for j, region in enumerate(all_regions):
            if is_in_region(x[i, :], region[0]):
                x_new[i, j] = d[j]
    return x_new


class AadIsolationForest(object):

    def __init__(self, n_estimators=10, max_samples=100,
                 score_type=IFOR_SCORE_TYPE_INV_PATH_LEN, random_state=None,
                 add_leaf_nodes_only=False):
        if random_state is None:
            self.random_state = np.random.RandomState(42)
        else:
            self.random_state = random_state

        self.n_estimators = n_estimators
        self.max_samples = max_samples

        self.score_type = score_type
        if not (self.score_type == IFOR_SCORE_TYPE_INV_PATH_LEN or
                        self.score_type == IFOR_SCORE_TYPE_INV_PATH_LEN_EXP or
                        self.score_type == IFOR_SCORE_TYPE_CONST or
                        self.score_type == IFOR_SCORE_TYPE_NEG_PATH_LEN):
            raise NotImplementedError("score_type %d not implemented!" % self.score_type)

        self.add_leaf_nodes_only = add_leaf_nodes_only

        self.clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                   random_state=self.random_state)

        # store all regions grouped by tree
        self.regions_in_forest = None

        # store all regions in a flattened list (ungrouped)
        self.all_regions = None

        # store maps of node index to region index for all trees
        self.all_node_regions = None

        # scores for each region
        self.d = None

        # samples for each region
        self.node_samples = None

        # fraction of instances in each region
        self.frac_insts = None

        # node weights learned through weak-supervision
        self.w = None

    def fit(self, x):
        self.clf.fit(x)
        # print len(clf.estimators_)
        # print type(clf.estimators_[0].tree_)

        self.regions_in_forest = []
        self.all_regions = []
        self.all_node_regions = []
        region_id = 0
        for i in range(len(self.clf.estimators_)):
            regions = self.extract_leaf_regions_from_tree(self.clf.estimators_[i],
                                                          self.add_leaf_nodes_only)
            self.regions_in_forest.append(regions)
            self.all_regions.extend(regions)
            node_regions = {}
            for region in regions:
                node_regions[region.node_id] = region_id
                region_id += 1  # this will monotonously increase across trees
            self.all_node_regions.append(node_regions)
            # print "%d, #nodes: %d" % (i, len(regions))
        self.d, self.node_samples, self.frac_insts = self.get_region_scores(self.all_regions)

    def extract_leaf_regions_from_tree(self, tree, add_leaf_nodes_only=False):
        """Extracts leaf regions from decision tree.

        Returns each decision path as array of strings representing
        node comparisons.

        Args:
            tree: sklearn.tree
                A trained decision tree.
            add_leaf_nodes_only: bool
                whether to extract only leaf node regions or include 
                internal node regions as well

        Returns: list of
        """

        add_intermediate_nodes = not add_leaf_nodes_only

        left = tree.tree_.children_left
        right = tree.tree_.children_right
        features = tree.tree_.feature
        threshold = tree.tree_.threshold
        node_samples = tree.tree_.n_node_samples

        # value = tree.tree_.value

        full_region = {}
        for fidx in range(tree.tree_.n_features):
            full_region[fidx] = (-np.inf, np.inf)

        regions = []

        def recurse(left, right, features, threshold, node, region, path_length=0):

            if left[node] == -1 and right[node] == -1:
                # we have reached a leaf node
                # print region
                regions.append(RegionData(deepcopy(region), path_length, node,
                                          self._average_path_length(node_samples[node]),
                                          node_samples[node]))
                return
            elif left[node] == -1 or right[node] == -1:
                print "dubious node..."

            feature = features[node]

            if add_intermediate_nodes and node != 0:
                regions.append(RegionData(deepcopy(region), path_length, node,
                                          self._average_path_length(node_samples[node]),
                                          node_samples[node]))

            if left[node] != -1:
                # make a copy to send down the next node so that
                # the previous value is unchanged when we backtrack.
                new_region = deepcopy(region)
                new_region[feature] = (new_region[feature][0], min(new_region[feature][1], threshold[node]))
                recurse(left, right, features, threshold, left[node], new_region, path_length + 1)

            if right[node] != -1:
                # make a copy for the reason mentioned earlier.
                new_region = deepcopy(region)
                new_region[feature] = (max(new_region[feature][0], threshold[node]), new_region[feature][1])
                recurse(left, right, features, threshold, right[node], new_region, path_length + 1)

        recurse(left, right, features, threshold, 0, full_region)
        return regions

    def _average_path_length(self, n_samples_leaf):
        """ The average path length in a n_samples iTree, which is equal to
        the average path length of an unsuccessful BST search since the
        latter has the same structure as an isolation tree.
        Parameters
        ----------
        n_samples_leaf : array-like of shape (n_samples, n_estimators), or int.
            The number of training samples in each test sample leaf, for
            each estimators.

        Returns
        -------
        average_path_length : array, same shape as n_samples_leaf

        """
        if n_samples_leaf <= 1:
            return 1.
        else:
            return 2. * (np.log(n_samples_leaf) + 0.5772156649) - 2. * (
                n_samples_leaf - 1.) / n_samples_leaf

    def decision_path_full(self, x, tree):
        """Returns the node ids of all nodes from root to leaf for each sample (row) in x
        
        Args:
            x: numpy.ndarray
            tree: fitted decision tree
        
        Returns: list of length x.shape[0]
            list of lists
        """

        left = tree.tree_.children_left
        right = tree.tree_.children_right
        features = tree.tree_.feature
        threshold = tree.tree_.threshold

        def path_recurse(x, left, right, features, threshold, node, path_nodes):
            """Returns the node ids of all nodes that x passes through from root to leaf
            
            Args:
                x: numpy.array
                    a single instance
                path_nodes: list
            """

            if left[node] == -1 and right[node] == -1:
                # reached a leaf
                return
            else:
                feature = features[node]
                if x[feature] <= threshold[node]:
                    next_node = left[node]
                else:
                    next_node = right[node]
                path_nodes.append(next_node)
                path_recurse(x, left, right, features, threshold, next_node, path_nodes)

        n = x.shape[0]
        all_path_nodes = []
        for i in xrange(n):
            path_nodes = []
            path_recurse(x[i, :], left, right, features, threshold, 0, path_nodes)
            all_path_nodes.append(path_nodes)
        return all_path_nodes

    def decision_path_leaf(self, x, tree):
        n = x.shape[0]
        all_path_nodes = []

        # get all leaf nodes
        node_idxs = tree.apply(x)
        # logger.debug("node_idxs:\n%s" % str(node_idxs))

        for j in range(n):
            all_path_nodes.append([node_idxs[j]])

        return all_path_nodes

    def get_decision_path(self, x, tree):
        if self.add_leaf_nodes_only:
            return self.decision_path_leaf(x, tree)
        else:
            return self.decision_path_full(x, tree)

    def decision_paths(self, x):
        all_decision_paths = []
        for tree in self.clf.estimators_:
            paths = self.decision_path_full(x, tree)
            all_decision_paths.append(paths)
        return all_decision_paths

    def get_region_scores(self, all_regions):
        d = np.zeros(len(all_regions))
        node_samples = np.zeros(len(all_regions))
        frac_insts = np.zeros(len(all_regions))
        for i, region in enumerate(all_regions):
            node_samples[i] = region.node_samples
            frac_insts[i] = region.node_samples * 1.0 / self.max_samples
            if self.score_type == IFOR_SCORE_TYPE_INV_PATH_LEN:
                d[i] = 1. / region.path_length
            elif self.score_type == IFOR_SCORE_TYPE_INV_PATH_LEN_EXP:
                d[i] = 2 ** -region.path_length  # used this to run the first batch
            elif self.score_type == IFOR_SCORE_TYPE_CONST:
                d[i] = -1
            elif self.score_type == IFOR_SCORE_TYPE_NEG_PATH_LEN:
                d[i] = -region.path_length
            else:
                # if self.score_type == IFOR_SCORE_TYPE_NORM:
                raise NotImplementedError("score_type %d not implemented!" % self.score_type)
                # d[i] = frac_insts[i]  # RPAD-ish
                # depth = region.path_length - 1
                # node_samples_avg_path_length = region.score
                # d[i] = (
                #            depth + node_samples_avg_path_length
                #        ) / (self.n_estimators * self._average_path_length(self.clf._max_samples))
        return d, node_samples, frac_insts

    def get_score(self, x, w):
        if self.score_type == IFOR_SCORE_TYPE_INV_PATH_LEN or \
                        self.score_type == IFOR_SCORE_TYPE_INV_PATH_LEN_EXP or \
                        self.score_type == IFOR_SCORE_TYPE_CONST or \
                        self.score_type == IFOR_SCORE_TYPE_NEG_PATH_LEN:
            return x.dot(w)
        else:
            raise NotImplementedError("score_type %d not implemented!" % self.score_type)

    def decision_function(self, x):
        return self.clf.decision_function(x)

    def transform_to_region_features(self, x, dense=True):
        """ Transforms matrix x to features from isolation forest

        :param x: np.ndarray
            Input data in original feature space
        :param dense: bool
            Whether to return a dense matrix or sparse. The number
            of features in isolation forest correspond to the nodes
            which might be thousands in number. However, each instance
            (row) in x will have only as many non-zero values as the
            number of trees -- which is *much* smaller than the number
            of nodes.
        :return:
        """
        if dense:
            return self.transform_to_region_features_dense(x)
        else:
            return self.transform_to_region_features_sparse_batch(x)

    def transform_to_region_features_dense(self, x):
        # return transform_features(x, self.all_regions, self.d)
        x_new = np.zeros(shape=(x.shape[0], len(self.d)), dtype=float)
        self._transform_to_region_features_with_lookup(x, x_new)
        return x_new

    def transform_to_region_features_sparse(self, x):
        # return transform_features(x, self.all_regions, self.d)
        x_new = lil_matrix((x.shape[0], len(self.d)), dtype=float)
        self._transform_to_region_features_with_lookup(x, x_new)
        return x_new.tocsr()

    def get_region_score_for_instance_transform(self, region_id, norm_factor=1.0):
        if self.score_type == IFOR_SCORE_TYPE_CONST:
            return self.d[region_id]
        else:
            return self.d[region_id] / norm_factor

    def transform_to_region_features_sparse_batch(self, x):
        """ Transforms from original feature space to IF node space
        
        The conversion to sparse vectors seems to take a lot of intermediate
        memory in python. This is why we are converting the vectors in smaller
        batches. The transformation is a one-time task, hence not a concern in 
        most cases.
        
        :param x: 
        :return: 
        """
        # logger.debug("transforming to IF feature space...")
        n = x.shape[0]
        m = len(self.d)
        batch_size = 10000
        start_batch = 0
        end_batch = min(start_batch + batch_size, n)
        x_new = csr_matrix((0, m), dtype=float)
        while start_batch < end_batch:
            starttime = timer()
            x_tmp = matrix(x[start_batch:end_batch, :], ncol=x.shape[1])
            x_tmp_new = lil_matrix((end_batch - start_batch, m), dtype=x_new.dtype)
            for i, tree in enumerate(self.clf.estimators_):
                n_tmp = x_tmp.shape[0]
                node_regions = self.all_node_regions[i]
                tree_paths = self.get_decision_path(x_tmp, tree)
                for j in xrange(n_tmp):
                    k = len(tree_paths[j])
                    for node_idx in tree_paths[j]:
                        region_id = node_regions[node_idx]
                        x_tmp_new[j, region_id] = self.get_region_score_for_instance_transform(region_id, k)
            if n >= 100000:
                endtime = timer()
                tdiff = difftime(endtime, starttime, units="secs")
                logger.debug("processed %d/%d (%f); batch %d in %f sec(s)" %
                             (end_batch + 1, n, (end_batch + 1)*1./n, batch_size, tdiff))
            x_new = vstack([x_new, x_tmp_new.tocsr()])
            start_batch = end_batch
            end_batch = min(start_batch + batch_size, n)
        return x_new

    def _transform_to_region_features_with_lookup(self, x, x_new):
        """ Transforms from original feature space to IF node space
        
        NOTE: This has been deprecated. Will be removed in future.
        
        Performs the conversion tree-by-tree. Even with batching by trees,
        this requires a lot of intermediate memory. Hence we do not use this method...
        
        :param x: 
        :param x_new: 
        :return: 
        """
        starttime = timer()
        n = x_new.shape[0]
        for i, tree in enumerate(self.clf.estimators_):
            node_regions = self.all_node_regions[i]
            for j in range(n):
                tree_paths = self.get_decision_path(matrix(x[j, :], nrow=1), tree)
                k = len(tree_paths[0])
                for node_idx in tree_paths[0]:
                    region_id = node_regions[node_idx]
                    x_new[j, region_id] = self.get_region_score_for_instance_transform(region_id, k)
                if j >= 100000:
                    if j % 20000 == 0:
                        endtime = timer()
                        tdiff = difftime(endtime, starttime, units="secs")
                        logger.debug("processed %d/%d trees, %d/%d (%f) in %f sec(s)" %
                                     (i, len(self.clf.estimators_), j + 1, n, (j + 1)*1./n, tdiff))

    def get_tau_ranked_instance(self, x, w, tau_rank):
        s = x.dot(w)
        ps = order(s, decreasing=True)[tau_rank]
        return matrix(x[ps, :], nrow=1)

    def get_aatp_quantile(self, x, w, topK):
        s = x.dot(w)
        return quantile(s, (1.0 - (topK * 1.0 / float(nrow(x)))) * 100.0)

    def if_aad_weight_update(self, w, x, y, hf, w_prior, opts, tau_rel=False, linear=True):
        n = x.shape[0]
        bt = get_budget_topK(n, opts)

        qval = self.get_aatp_quantile(x, w, bt.topK)

        x_tau = None
        if tau_rel:
            x_tau = self.get_tau_ranked_instance(x, w, bt.topK)
            # logger.debug("x_tau:")
            # logger.debug(to_dense_mat(x_tau))

        def if_f(w, x, y):
            if linear:
                return if_aad_loss_linear(w, x, y, qval, x_tau=x_tau,
                                          Ca=opts.Ca, Cn=opts.Cn, Cx=opts.Cx,
                                          withprior=opts.withprior, w_prior=w_prior,
                                          sigma2=opts.priorsigma2)
            else:
                return if_aad_loss_exp(w, x, y, qval, x_tau=x_tau,
                                       Ca=opts.Ca, Cn=opts.Cn, Cx=opts.Cx,
                                       withprior=opts.withprior, w_prior=w_prior,
                                       sigma2=opts.priorsigma2)

        def if_g(w, x, y):
            if linear:
                return if_aad_loss_gradient_linear(w, x, y, qval, x_tau=x_tau,
                                                   Ca=opts.Ca, Cn=opts.Cn, Cx=opts.Cx,
                                                   withprior=opts.withprior, w_prior=w_prior,
                                                   sigma2=opts.priorsigma2)
            else:
                return if_aad_loss_gradient_exp(w, x, y, qval, x_tau=x_tau,
                                                Ca=opts.Ca, Cn=opts.Cn, Cx=opts.Cx,
                                                withprior=opts.withprior, w_prior=w_prior,
                                                sigma2=opts.priorsigma2)

        w_new = sgd(w, x[hf, :], y[hf], if_f, if_g,
                    learning_rate=0.001, max_epochs=1000, eps=1e-5,
                    shuffle=True, rng=self.random_state)
        w_new = w_new / np.sqrt(w_new.dot(w_new))
        return w_new

    def get_uniform_weights(self, m=None):
        if m is None:
            m = len(self.d)
        w_unif = np.ones(m, dtype=float)
        w_unif = w_unif / np.sqrt(w_unif.dot(w_unif))
        # logger.debug("w_prior:")
        # logger.debug(w_unif)
        return w_unif

    def order_by_score(self, x, w=None):
        if w is None:
            anom_score = self.get_score(x, self.w)
        else:
            anom_score = self.get_score(x, w)
        return order(anom_score, decreasing=True)

    def aad_ensemble(self, ensemble, opts):

        if opts.budget == 0:
            return None

        x = ensemble.scores
        y = ensemble.labels

        n, m = x.shape
        bt = get_budget_topK(n, opts)

        metrics = get_alad_metrics_structure(opts.budget, opts)
        ha = []
        hn = []
        xis = []

        qstate = Query.get_initial_query_state(opts.qtype, opts=opts, qrank=bt.topK)

        metrics.all_weights = np.zeros(shape=(opts.budget, m))

        w_unif_prior = self.get_uniform_weights(m)
        if self.w is None:
            self.w = w_unif_prior

        for i in range(bt.budget):

            starttime_iter = timer()

            # save the weights in each iteration for later analysis
            metrics.all_weights[i, :] = self.w
            metrics.queried = xis  # xis keeps growing with each feedback iteration

            order_anom_idxs = self.order_by_score(x)

            xi = qstate.get_next_query(maxpos=n, ordered_indexes=order_anom_idxs,
                                       queried_items=xis,
                                       x=x, lbls=y,
                                       w=self.w, hf=append(ha, hn),
                                       remaining_budget=opts.budget - i)
            # logger.debug("xi: %d" % (xi,))
            xis.append(xi)

            if opts.single_inst_feedback:
                # Forget the previous feedback instances and
                # use only the current feedback for weight updates
                ha = []
                hn = []

            if y[xi] == 1:
                ha.append(xi)
            else:
                hn.append(xi)

            qstate.update_query_state(rewarded=(y[xi] == 1))

            if opts.batch:
                # Use the original (uniform) weights as prior
                self.w = w_unif_prior
                hf = np.arange(i)
                ha = hf[np.where(y[hf] == 1)[0]]
                hn = hf[np.where(y[hf] == 0)[0]]

            if opts.unifprior:
                w_prior = w_unif_prior
            else:
                w_prior = self.w

            tau_rel = opts.constrainttype == AAD_CONSTRAINT_TAU_INSTANCE
            if opts.update_type == AAD_IFOREST:
                self.w = self.if_aad_weight_update(self.w, x, y, hf=append(ha, hn),
                                              w_prior=w_prior, opts=opts, tau_rel=tau_rel)
            elif opts.update_type == ATGP_IFOREST:
                w_soln = weight_update_iter_grad(ensemble.scores, ensemble.labels,
                                                 hf=append(ha, hn),
                                                 Ca=opts.Ca, Cn=opts.Cn, Cx=opts.Cx,
                                                 topK=bt.topK, max_iters=1000)
                self.w = w_soln.w
            else:
                raise ValueError("Invalid weight update for IForest: %d" % opts.update_type)
            # logger.debug("w_new:")
            # logger.debug(w_new)

            if np.mod(i, 1) == 0:
                endtime_iter = timer()
                tdiff = difftime(endtime_iter, starttime_iter, units="secs")
                logger.debug("Completed [%s] fid %d rerun %d feedback %d in %f sec(s)" %
                             (opts.dataset, opts.fid, opts.runidx, i, tdiff))

        return metrics

    def run_aad(self, samples, labels, scores, w, opts):

        starttime_feedback = timer()

        agg_scores = scores.dot(w)
        ensemble = Ensemble(samples, labels, scores, w,
                            agg_scores=agg_scores, original_indexes=np.arange(samples.shape[0]),
                            auc=0.0, model=None)

        metrics = alad_ensemble(ensemble, opts)

        num_seen = None
        num_seen_baseline = None
        queried_indexes = None
        queried_indexes_baseline = None

        if metrics is not None:
            save_alad_metrics(metrics, opts)
            num_seen, num_seen_baseline, queried_indexes, queried_indexes_baseline = \
                summarize_ensemble_num_seen(ensemble, metrics, fid=opts.fid)
            logger.debug("baseline: \n%s" % str([v for v in num_seen_baseline[0, :]]))
            logger.debug("num_seen: \n%s" % str([v for v in num_seen[0, :]]))

        endtime_feedback = timer()

        tdiff = difftime(endtime_feedback, starttime_feedback, units="secs")
        logger.debug("Processed [%s] file %d, auc: %f, time: %f sec(s); completed at %s" %
                     (opts.dataset, opts.fid, ensemble.auc, tdiff, endtime_feedback))

        return num_seen, num_seen_baseline, queried_indexes, queried_indexes_baseline

    def save_alad_metrics(self, metrics, opts):
        cansave = (opts.resultsdir != "" and os.path.isdir(opts.resultsdir))
        if cansave:
            save(metrics, filepath=opts.get_metrics_path())

    def load_alad_metrics(self, opts):
        metrics = None
        fpath = opts.get_metrics_path()
        canload = (opts.resultsdir != "" and os.path.isfile(fpath))
        if canload:
            # print "Loading metrics" + fpath
            metrics = load(fpath)
        else:
            print "Cannot load " + fpath
        return metrics


def write_sparsemat_to_file(fname, X, fmt='%.18e', delimiter=','):
    if isinstance(X, np.ndarray):
        np.savetxt(fname, X, fmt='%3.2f', delimiter=",")
    elif isinstance(X, csr_matrix):
        f = open(fname, 'w')
        for i in range(X.shape[0]):
            a = X[i, :].toarray()[0]
            f.write(delimiter.join([fmt % v for v in a]))
            f.write(os.linesep)
            if (i + 1) % 10 == 0:
                f.flush()
        f.close()
    else:
        raise ValueError("Invalid matrix type")


def get_num_batches(n, batch_size):
    return int(round((n + batch_size * 0.5) / batch_size))


def get_sgd_batch(x, y, i, batch_size, shuffled_idxs=None):
    s = i * batch_size
    e = min(x.shape[0], (i + 1) * batch_size)
    if shuffled_idxs is None:
        idxs = np.arange(s, e)
    else:
        idxs = shuffled_idxs[np.arange(s, e)]
    return matrix(x[idxs, :], ncol=x.shape[1]), y[idxs]


def sgd(w0, x, y, f, grad, learning_rate=0.01,
        batch_size=100, max_epochs=100, eps=1e-6, shuffle=False, rng=None):
    n = x.shape[0]
    n_batches = get_num_batches(n, batch_size)
    w = w0
    epoch_losses = np.zeros(max_epochs, dtype=float)
    epoch = 0
    w_best = w0
    loss_best = np.inf
    if shuffle:
        shuffled_idxs = np.arange(n)
        if rng is None:
            np.random.shuffle(shuffled_idxs)
        else:
            rng.shuffle(shuffled_idxs)
    else:
        shuffled_idxs = None
    while epoch < max_epochs:
        losses = np.zeros(n_batches, dtype=float)
        for i in range(n_batches):
            xi, yi = get_sgd_batch(x, y, i, batch_size, shuffled_idxs=shuffled_idxs)
            w -= learning_rate * grad(w, xi, yi)
            losses[i] = f(w, xi, yi)
        loss = np.mean(losses)
        epoch_losses[epoch] = loss
        if loss < loss_best:
            # pocket algorithm
            w_best = w
            loss_best = loss
        if loss < eps:
            break
        epoch += 1
    # print epoch
    # logger.debug("net losses:")
    # logger.debug(epoch_losses[0:epoch])
    # logger.debug("best loss: %f" % loss_best)
    return w_best


def get_aad_iforest_args(dataset="", budget=1, reruns=1, log_file=""):

    debug_args = [
        "--dataset=%s" % dataset,
        "--log_file=",
        "--querytype=%d" % QUERY_DETERMINISIC,
        "--inferencetype=%d" % AAD_IFOREST,
        # "--constrainttype=%d" % AAD_CONSTRAINT_TAU_INSTANCE,
        "--constrainttype=%d" % AAD_CONSTRAINT_NONE,
        "--withprior",
        "--unifprior",
        "--debug",
        "--sigma2=0.1",
        "--Ca=100",
        "--Cx=0.001",
        "--budget=%d" % budget,
        "--reruns=%d" % reruns,
        "--runtype=%s" % ("multi" if reruns > 1 else "simple")
    ]

    # the reason to use 'debug=True' below is to have the arguments
    # read from the debug_args list and not commandline.
    args = get_command_args(debug=True, debug_args=debug_args)
    args.log_file = log_file
    configure_logger(args)
    return args
