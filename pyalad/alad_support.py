from ensemble_support import *
from alad_simple import *
from weight_inference import *
from aatp_iterative_gradient import *
from query_model import *


class MetricsStructure(object):
    def __init__(self, train_aucs=None, test_aucs=None, train_precs=None, test_precs=None,
                 train_aprs=None, test_aprs=None, train_n_at_top=None, test_n_at_top=None,
                 all_weights=None, queried=None):
        self.train_aucs = train_aucs
        self.test_aucs = test_aucs
        self.train_precs = train_precs
        self.test_precs = test_precs
        self.train_aprs = train_aprs
        self.test_aprs = test_aprs
        self.train_n_at_top = train_n_at_top
        self.test_n_at_top = test_n_at_top
        self.all_weights = all_weights
        self.queried = queried


def get_alad_metrics_structure(budget, opts):
    metrics = MetricsStructure(
        train_aucs=np.zeros(shape=(1, budget)),
        # for precision@k first two columns are fid,k
        train_precs=[],
        train_aprs=np.zeros(shape=(1, budget)),
        train_n_at_top=[],
        all_weights=[],
        queried=[]
    )
    for k in range(len(opts.precision_k)):
        metrics.train_precs.append(np.zeros(shape=(1, budget)))
        metrics.train_n_at_top.append(np.zeros(shape=(1, budget)))
    return metrics


def save_alad_metrics(metrics, opts):
    cansave = (opts.resultsdir != "" and os.path.isdir(opts.resultsdir))
    if cansave:
        save(metrics, filepath=opts.get_metrics_path())


def load_alad_metrics(opts):
    metrics = None
    fpath = opts.get_metrics_path()
    canload = (opts.resultsdir != "" and os.path.isfile(fpath))
    if canload:
        # print "Loading metrics" + fpath
        metrics = load(fpath)
    else:
        print "Cannot load " + fpath
    return metrics


class MetricsCollection(object):
    def __init__(self, fids, runidxs, metrics):
        self.fids = fids
        self.runidxs = runidxs
        self.metrics = metrics


def consolidate_alad_metrics(fids, runidxs, opts):
    """

    :param fids:
    :param runidxs:
    :param opts:
    :return: MetricsCollection
    """
    metrics_struct = MetricsCollection(fids=fids, runidxs=runidxs, metrics=[])
    for i in range(len(fids)):
        fmetrics = []
        for j in range(len(runidxs)):
            opts.set_multi_run_options(fids[i], runidxs[j])
            fmetrics.append(load_alad_metrics(opts))
        metrics_struct.metrics.append(fmetrics)
    return metrics_struct


class SequentialResults(object):
    def __init__(self, num_seen, num_seen_baseline,
                 true_queried_indexes=None, true_queried_indexes_baseline=None):
        self.num_seen = num_seen
        self.num_seen_baseline = num_seen_baseline
        self.true_queried_indexes = true_queried_indexes
        self.true_queried_indexes_baseline = true_queried_indexes_baseline


def summarize_ensemble_num_seen(ensemble, metrics, fid=0, runidx=0):
    """
    IMPORTANT: returned queried_indexes and queried_indexes_baseline are 1-indexed (NOT 0-indexed)
    """
    nqueried = len(metrics.queried)
    num_seen = np.zeros(shape=(1, nqueried + 2))
    num_seen_baseline = np.zeros(shape=(1, nqueried + 2))

    num_seen[0, 0:2] = [fid, runidx]
    num_seen[0, 2:(num_seen.shape[1])] = np.cumsum(ensemble.labels[metrics.queried])

    qlbls = ensemble.labels[ensemble.ordered_anom_idxs[0:nqueried]]
    num_seen_baseline[0, 0:2] = [fid, runidx]
    num_seen_baseline[0, 2:(num_seen_baseline.shape[1])] = np.cumsum(qlbls)

    # the ensembles store samples in sorted order of default anomaly
    # scores. The corresponding indexes are stored in ensemble.original_indexes
    true_queried_indexes = np.zeros(shape=(1, nqueried + 2))
    true_queried_indexes[0, 0:2] = [fid, runidx]
    # Note: make the queried indexes relative 1 (NOT zero)
    true_queried_indexes[0, 2:(true_queried_indexes.shape[1])] = ensemble.original_indexes[metrics.queried] + 1

    # the ensembles store samples in sorted order of default anomaly
    # scores. The corresponding indexes are stored in ensemble.original_indexes
    true_queried_indexes_baseline = np.zeros(shape=(1, nqueried + 2))
    true_queried_indexes_baseline[0, 0:2] = [fid, runidx]
    # Note: make the queried indexes relative 1 (NOT zero)
    true_queried_indexes_baseline[0, 2:(true_queried_indexes_baseline.shape[1])] = \
        ensemble.original_indexes[np.arange(nqueried)] + 1

    return num_seen, num_seen_baseline, true_queried_indexes, true_queried_indexes_baseline


def summarize_alad_metrics(ensembles, metrics_struct):
    nqueried = len(metrics_struct.metrics[0][0].queried)
    num_seen = np.zeros(shape=(0, nqueried+2))
    num_seen_baseline = np.zeros(shape=(0, nqueried+2))
    true_queried_indexes = np.zeros(shape=(0, nqueried+2))
    true_queried_indexes_baseline = np.zeros(shape=(0, nqueried + 2))
    for i in range(len(metrics_struct.metrics)):
        # file level
        submetrics = metrics_struct.metrics[i]
        subensemble = ensembles[i]
        for j in range(len(submetrics)):
            # rerun level
            queried = submetrics[j].queried
            lbls = subensemble[j].labels

            nseen = np.zeros(shape=(1, nqueried+2))
            nseen[0, 0:2] = [metrics_struct.fids[i], metrics_struct.runidxs[j]]
            nseen[0, 2:(nseen.shape[1])] = np.cumsum(lbls[queried])
            num_seen = rbind(num_seen, nseen)

            qlbls = subensemble[j].labels[subensemble[j].ordered_anom_idxs[0:nqueried]]
            nseen = np.zeros(shape=(1, nqueried+2))
            nseen[0, 0:2] = [metrics_struct.fids[i], metrics_struct.runidxs[j]]
            nseen[0, 2:(nseen.shape[1])] = np.cumsum(qlbls)
            num_seen_baseline = rbind(num_seen_baseline, nseen)

            # the ensembles store samples in sorted order of default anomaly
            # scores. The corresponding indexes are stored in ensemble.original_indexes
            t_idx = np.zeros(shape=(1, nqueried + 2))
            t_idx[0, 0:2] = [metrics_struct.fids[i], metrics_struct.runidxs[j]]
            t_idx[0, 2:(t_idx.shape[1])] = subensemble[j].original_indexes[queried]
            # Note: make the queried indexes realive 1 (NOT zero)
            true_queried_indexes = rbind(true_queried_indexes, t_idx + 1)

            # the ensembles store samples in sorted order of default anomaly
            # scores. The corresponding indexes are stored in ensemble.original_indexes
            b_idx = np.zeros(shape=(1, nqueried + 2))
            b_idx[0, 0:2] = [metrics_struct.fids[i], metrics_struct.runidxs[j]]
            b_idx[0, 2:(b_idx.shape[1])] = subensemble[j].original_indexes[np.arange(nqueried)]
            # Note: make the queried indexes realive 1 (NOT zero)
            true_queried_indexes_baseline = rbind(true_queried_indexes_baseline, b_idx + 1)

    return SequentialResults(num_seen=num_seen, num_seen_baseline=num_seen_baseline,
                             true_queried_indexes=true_queried_indexes,
                             true_queried_indexes_baseline=true_queried_indexes_baseline)


def save_alad_summary(alad_summary, opts):
    cansave = opts.resultsdir != "" and os.path.isdir(opts.resultsdir)
    if cansave:
        save(alad_summary, filepath=opts.get_metrics_summary_path())


def load_alad_summary(opts):
    alad_summary = None
    fpath = opts.get_metrics_summary_path()
    canload = opts.resultsdir != "" and os.path.isfile(fpath)
    if canload:
        alad_summary = load(fpath)
    else:
        print ("Cannot load %s" % fpath)
    return alad_summary


class Budget(object):
    def __init__(self, topK, budget):
        self.topK = topK
        self.budget = budget


def get_budget_topK(n, opts):
    # set topK as per tau or input topK
    topK = opts.topK
    if topK <= 0:
        topK = int(np.round(opts.tau * n))  # function of total number of instances
    budget = opts.budget
    if budget <= 0:
        budget = int(np.round(opts.tau * n))
    budget = min(opts.maxbudget, budget)
    return Budget(topK=topK, budget=budget)


def alad_ensemble(ensemble, opts):
    """Main procedure for ALAD

    Args:
        ensemble: ensemble_support.Ensemble
        opts: app_globals.Opts

    Returns: MetricsStructure
    """

    n, m = ensemble.scores.shape

    bt = get_budget_topK(n, opts)

    budget = bt.budget
    topK = bt.topK

    if budget < 1:
        logger.debug("completed %d because budget < 1" % (opts.fid,))
        return None

    metrics = get_alad_metrics_structure(budget, opts)

    ha = []
    hn = []
    xis = []
    qval = np.Inf
    qval_ranges = []
    qvals = []  # just to check the trend whether this increases or decreases across iterations

    qstate = Query.get_initial_query_state(opts.qtype, opts=opts, qrank=topK)

    metrics.all_weights = np.zeros(shape=(budget, m))
    detector_wts = ensemble.weights

    for i in range(budget):

        starttime_iter = timer()

        # save the weights in each iteration for later analysis
        metrics.all_weights[i, :] = detector_wts
        metrics.queried = xis  # xis keeps growing with each feedback iteration

        anom_score = ensemble.scores.dot(detector_wts)
        order_anom_idxs = order(anom_score, decreasing=True)

        if True:
            # gather AUC metrics
            metrics.train_aucs[0, i] = fn_auc(cbind(ensemble.labels, -anom_score))

            # gather Precision metrics
            prec = fn_precision(cbind(ensemble.labels, -anom_score), opts.precision_k)
            metrics.train_aprs[0, i] = prec[len(opts.precision_k) + 1]
            train_n_at_top = get_anomalies_at_top(-anom_score, ensemble.labels, opts.precision_k)
            for k in range(len(opts.precision_k)):
                metrics.train_precs[k][0, i] = prec[k]
                metrics.train_n_at_top[k][0, i] = train_n_at_top[k]

        xi = qstate.get_next_query(maxpos=n, ordered_indexes=order_anom_idxs,
                                   queried_items=xis,
                                   x=ensemble.scores, lbls=ensemble.labels,
                                   w=detector_wts, hf=append(ha, hn),
                                   remaining_budget=opts.budget - i)
        # logger.debug("xi: %d" % (xi,))
        xis.append(xi)

        if opts.single_inst_feedback:
            # Forget the previous feedback instances and
            # use only the current feedback for weight updates
            ha = []
            hn = []
        if ensemble.labels[xi] == 1:
            ha.append(xi)
        else:
            hn.append(xi)

        qstate.update_query_state(rewarded=(ensemble.labels[xi] == 1))

        if opts.batch:
            # Use the original (uniform) weights as prior
            detector_wts = rep(1. / np.sqrt(m), m)
            hf = np.arange(i)
            ha = hf[np.where(ensemble.labels[hf] == 1)[0]]
            hn = hf[np.where(ensemble.labels[hf] == 0)[0]]

        if opts.unifprior:
            w_prior = rep(1. / np.sqrt(m), m)
        else:
            w_prior = detector_wts

        if True:
            # for debug, log range of scores
            qval_ranges.append(get_score_ranges(ensemble.scores, detector_wts))

        topK = bt.topK
        if (opts.update_type == AAD_UPD_TYPE or
                    opts.update_type == AAD_SLACK_CONSTR_UPD_TYPE):
            if i == 0 and opts.random_instance_at_start:
                topK = np.random.random_integers(1, ensemble.scores.shape[0], 1)[0]
                # logger.debug("random inst-index: %d" % topK)
            qval = get_aatp_quantile(x=ensemble.scores, w=detector_wts, topK=topK)
            qvals.append(qval)

        if opts.update_type == SIMPLE_UPD_TYPE:
            # The simple online weight update
            # enforces ||w||=1 constraint
            detector_wts = weight_update_online_simple(ensemble.scores, ensemble.labels, hf=append(ha, hn),
                                                       w=detector_wts,
                                                       nu=opts.nu, Ca=opts.Ca, Cn=opts.Cn,
                                                       sigma2=opts.priorsigma2,
                                                       relativeto=opts.relativeto,
                                                       tau_anomaly=opts.tau, tau_nominal=opts.tau_nominal)
        elif opts.update_type == SIMPLE_UPD_TYPE_R_OPTIM:
            raise NotImplementedError("Update type %d not implemented!" % (opts.update_type,))
        elif (opts.update_type == AAD_UPD_TYPE or
                opts.update_type == AAD_SLACK_CONSTR_UPD_TYPE):

            # AATP weight update
            w_soln = weight_update_aatp_slack_pairwise_constrained(
                ensemble.scores, ensemble.labels,
                hf=append(ha, hn),
                w=detector_wts, qval=qval,
                Ca=opts.Ca, Cn=opts.Cn, Cx=opts.Cx,
                withprior=opts.withprior,
                w_prior=w_prior,
                w_old=detector_wts,
                sigma2=opts.priorsigma2,
                pseudoanomrank=topK,
                pseudoanomrank_always=opts.pseudoanomrank_always,
                order_by_violated=opts.orderbyviolated,
                ignore_aatp_loss=opts.ignoreAATPloss,
                random_instance_at_start=opts.random_instance_at_start,
                constraint_type=opts.constrainttype,
                max_anomalies_in_constraint_set=opts.max_anomalies_in_constraint_set,
                max_nominals_in_constraint_set=opts.max_nominals_in_constraint_set,
                optimlib=opts.optimlib)

            if w_soln.success:
                detector_wts = w_soln.w
            else:
                logger.warning("Warning: Error in optimization for iter %d" % (i,))
                # retain the previous weights
        elif opts.update_type == AAD_ITERATIVE_GRAD_UPD_TYPE:
            # enforces \sum{w_i}=1 constraint
            w_soln = weight_update_iter_grad(ensemble.scores, ensemble.labels,
                                             hf=append(ha, hn),
                                             Ca=opts.Ca, Cn=opts.Cn, Cx=opts.Cx, topK=topK, max_iters=1000)
            #logger.debug("Iter: %d; old loss: %f; loss: %f; del_loss: %f" %
            #             (w_soln.tries, w_soln.loss_old, w_soln.loss, abs(w_soln.loss - w_soln.loss_old)))
            detector_wts = w_soln.w
        elif opts.update_type == SIMPLE_PAIRWISE:
            w_soln = weight_update_simple_pairwise(ensemble.scores, ensemble.labels,
                                                   hf=append(ha, hn),
                                                   w=detector_wts, w_prior=w_prior,
                                                   Ca=opts.Ca, Cn=opts.Cn,
                                                   sigma2=opts.priorsigma2,
                                                   topK=topK)
            detector_wts = w_soln.w
            # logger.debug(detector_wts)
        else:
            # older types of updates, not used anymore...
            raise ValueError("Invalid weight update specified!")

        if np.mod(i, 1) == 0:
            endtime_iter = timer()
            tdiff = difftime(endtime_iter, starttime_iter, units="secs")
            logger.debug("Completed [%s] fid %d rerun %d feedback %d in %f sec(s)" %
                         (opts.dataset, opts.fid, opts.runidx, i, tdiff))
    # logger.debug("[%s] fid %d rerun %d\nqvals: %s" % (opts.dataset, opts.fid, opts.runidx,
    #                                                   ",".join([str(v) for v in qvals])))
    # logger.debug("[%s] fid %d rerun %d\nscore_ranges: %s" % (opts.dataset, opts.fid, opts.runidx,
    #                                                          ",".join([",".join([("%0.3f" % v) for v in arr]) for arr in qval_ranges])))

    return metrics


def run_alad_simple(samples, labels, opts, rnd_seed=0):

    ensemblemanager = EnsembleManager.get_ensemble_manager(opts)

    np.random.seed(rnd_seed)

    ensemble = ensemblemanager.load_data(samples, labels, opts)

    starttime_feedback = timer()

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


def run_alad_multi(samples, labels, opts, rnd_seed=0):

    ensemblemanager = EnsembleManager.get_ensemble_manager(opts)
    all_num_seen = None
    all_num_seen_baseline = None
    all_queried_indexes = None
    all_queried_indexes_baseline = None

    runidxs = opts.get_runidxs()
    for runidx in runidxs:
        starttime_feedback = timer()

        opts.set_multi_run_options(opts.fid, runidx)

        zvars = get_zero_var_features(samples)
        if opts.original_dims and zvars is not None:
            # print(c("Zero variance features: ", zvars))
            # a = a[,-zvars]
            opts.exclude = zvars

        # print(sprintf("rnd_seed: %d, fid: %d, runidx:    %d", rnd_seed, fid, runidx))
        sub_rnd_seed = rnd_seed + (opts.fid*100) + runidx - 1
        np.random.seed(sub_rnd_seed)

        if opts.ensembletype == "regular" and not opts.is_simple_run():
            opts.scoresfile = os.path.join(opts.scoresdir,
                                           "%s_%d_%d.csv" % (opts.dataset, opts.fid, opts.runidx))

        ensemble = ensemblemanager.load_data(samples, labels, opts)

        # reset seed in case precached model is loaded.
        # this will make sure that the operations later will be
        # reproducible
        np.random.seed(sub_rnd_seed + 32767)

        metrics = alad_ensemble(ensemble, opts)
        if metrics is not None:
            save_alad_metrics(metrics, opts)

            num_seen, num_seen_baseline, queried_indexes, queried_indexes_baseline = \
                summarize_ensemble_num_seen(ensemble, metrics, fid=opts.fid, runidx=runidx)
            all_num_seen = rbind(all_num_seen, num_seen)
            all_num_seen_baseline = rbind(all_num_seen_baseline, num_seen_baseline)
            all_queried_indexes = rbind(all_queried_indexes, queried_indexes)
            all_queried_indexes_baseline = rbind(all_queried_indexes_baseline, queried_indexes_baseline)
            logger.debug("baseline: \n%s" % str([v for v in num_seen_baseline[0, :]]))
            logger.debug("num_seen: \n%s" % str([v for v in num_seen[0, :]]))

        endtime_feedback = timer()
        tdiff = difftime(endtime_feedback, starttime_feedback, units="secs")
        logger.debug("Processed [%s] file %d, auc: %f, time: %f sec(s); completed at %s" %
                     (opts.dataset, opts.fid, ensemble.auc, tdiff, endtime_feedback))

    return all_num_seen, all_num_seen_baseline, all_queried_indexes, all_queried_indexes_baseline


def alad(opts):

    if opts.is_simple_run():
        logger.debug("Simple run...")
        allsamples = [load_samples(opts.datafile, opts, fid=0)]
    else:
        logger.debug("Multi run...")
        allsamples = load_all_samples(opts.dataset, opts.filedir, opts.get_fids(), opts)

    logger.debug("loaded all %s samples..." % (opts.dataset,))

    all_num_seen = None
    all_num_seen_baseline = None
    all_queried_indexes = None
    all_queried_indexes_baseline = None
    for i in range(len(allsamples)):
        # args.sparsity = 0.0 # include all features
        opts.sparsity = None  # loda default d*(1-1/sqrt(d)) vectors will be zero

        fid = allsamples[i].fid
        samples = allsamples[i].fmat
        labels = allsamples[i].lbls

        rnd_seed = opts.randseed + fid

        opts.fid = fid

        bt = get_budget_topK(nrow(samples), opts)
        budget = bt.budget
        topK = bt.topK

        logger.debug("topK: %d, budget: %d, tau: %f" % (topK, budget, opts.tau))

        if opts.is_simple_run():
            num_seen_summary = run_alad_simple(samples, labels, opts, rnd_seed)
        else:
            num_seen_summary = run_alad_multi(samples, labels, opts, rnd_seed)

        all_num_seen = rbind(all_num_seen, num_seen_summary[0])
        all_num_seen_baseline = rbind(all_num_seen_baseline, num_seen_summary[1])
        all_queried_indexes = rbind(all_queried_indexes, num_seen_summary[2])
        all_queried_indexes_baseline = rbind(all_queried_indexes_baseline, num_seen_summary[3])

        logger.debug("completed %s fid %d" % (opts.dataset, fid,))

    return SequentialResults(num_seen=all_num_seen, num_seen_baseline=all_num_seen_baseline,
                             true_queried_indexes=all_queried_indexes,
                             true_queried_indexes_baseline=all_queried_indexes_baseline)


