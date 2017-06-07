from loda_support import *
from alad_simple import *
from weight_inference import *


class ActionValue(object):
    def __init__(self, action, value):
        self.action = action
        self.value = value


def get_next_query_and_utility(x=None, lbls=None,
                               w=None, hf=None, remaining_budget=0,
                               k=0, a=None, y=None, opts=None):
    #logger.debug("query search budget: %d, k: %d, a: %d, y: %d" % (remaining_budget, k, a, y))
    k = min(k, remaining_budget)

    hf_new = hf
    lbls_new = lbls
    w_new = w
    scores = x.dot(w)
    scores_new = scores

    scores_ecdf = ecdf(scores)
    scores_new_ecdf = scores_ecdf

    if False:
        # DEBUG only block
        ordered_indexes = order(scores_new, decreasing=True)
        p = p_logistic(x, ordered_indexes, tau=opts.tau)
        logger.debug(np.round(p[:, 0], 4))
        raise NotImplementedError("debug")
    
    if a is not None:
        hf_new = append(hf, [a])
        lbls_new[a] = y
        w_new = weight_update_online_simple(x, lbls_new, hf_new, w,
                                            nu=opts.nu, Ca=opts.Ca, Cn=opts.Cn,
                                            sigma2=opts.priorsigma2,
                                            relativeto=opts.relativeto,
                                            tau_anomaly=opts.tau, tau_nominal=opts.tau_nominal)
        scores_new = x.dot(w_new)
        scores_new_ecdf = ecdf(scores_new)
        R = compute_reward(x, w, hf, scores, w_new, hf_new, scores_new, opts,
                           scores_ecdf=scores_ecdf, scores_new_ecdf=scores_new_ecdf)
        logger.debug("inst: %d, R: %f, rem. budget: %d, k: %d" % (a, R, remaining_budget, k))
    else:
        R = 0
    
    if k <= 0 or remaining_budget <= 0:
        if k > remaining_budget:
            raise ValueError("warning: k > remaining_budget (k=%d, rem. budget=%d)" %
                             (k, remaining_budget))
        return ActionValue(action=a, value=R)
    
    Q = generate_query_candidates(x, hf_new, w_new, scores_new, remaining_budget, opts)

    A = []
    best_i = 0
    best_v = -np.Inf
    for i in range(len(Q)):
        uv_0 = get_next_query_and_utility(x, lbls_new, w_new, hf_new,
                                          remaining_budget - 1, k - 1, Q[i], 0, opts)
        uv_1 = get_next_query_and_utility(x, lbls_new, w_new, hf_new,
                                          remaining_budget - 1, k - 1, Q[i], 1, opts)
        p_anomaly_given_x = scores_new_ecdf(scores_new[Q[i]])
        v_i = (p_anomaly_given_x * uv_1.value) + ((1-p_anomaly_given_x) * uv_0.value)  # compute the value
        logger.debug("inst: %d, v: %f, v_0: %f, v_1: %f, p(y=1|x): %f" %
                     (Q[i], v_i, uv_0.value, uv_1.value, p_anomaly_given_x))
        A.append(ActionValue(action=Q[i], value=v_i))
        if v_i > best_v:
            best_i = i
            best_v = v_i
    if a is None:
        # only when called initially and not for intermittent stages
        logger.debug("hf: %s" % ",".join([str(i) for i in hf]))
        logger.debug("best inst: %d" % A[best_i].action)
    
    return A[best_i]


def compute_reward(x, w, hf, scores, w_new, hf_new, scores_new, opts,
                     scores_ecdf=None, scores_new_ecdf=None):
    n = nrow(x)

    #ordidxs = order(scores, decreasing = TRUE)
    #ordidxs_new = order(scores_new, decreasing = TRUE)

    # find difference in scores between top ranked 10% anomalies
    #reward = top_ranked_score_diff(scores, ordidxs, scores_new, ordidxs_new, opts.tau)

    voi = voi_loss(x, scores_new, hf_new, opts.tau) - voi_loss(x, scores, hf, opts.tau)
    #print voi_loss(x, scores_new, hf_new, opts.tau)
    #print voi_loss(x, scores, hf, opts.tau)
    reward = voi

    # find expected loss

    # find how well-separated are top ranked anomalies from others

    # find difference in entropy of the top ranked anomalies from nominals' entropy

    # find sum of absolute ranks of known anomalies --> minimize this

    return reward


def voi_loss(x, scores, hf, tau):
    # compute value of information (VOI)
    # VOI is defined as reduction in expected loss.
    # Here we will compute loss as in Culotta et al. (2014):
    # loss(P(y|x)) =
    #     1 - P(y=1|x) if rank(x) < n * tau
    #     1 - P(y=0|x) otherwise
    ordered_indexs = order(scores, decreasing=True)
    n = nrow(x)
    topn = n * tau
    p = p_logistic(x, ordered_indexs, tau)
    # logger.debug(p)
    loss = 0
    for i in range(n):
        item = ordered_indexs[i]
        if len(np.where(hf == item)[0]) > 0:
            p_y1 = p[item]    #scores_ecdf(scores[item])
            if i <= topn:
                loss += (1 - p_y1)
            else:
                loss += p_y1
    return loss / float(n-len(hf))


def generate_query_candidates(x, hf, w, scores, remaining_budget, opts):
    # generate one-step query candidates
    # with probability p_r = f(remaining_budget, max_budget) return candidates for exploration.
    # with probability 1-p_r return the top ranked instance.
    ordered_idxs = order(scores, decreasing = True)
    p_r = exploration_probability(remaining_budget, opts.budget)
    logger.debug("Explore prob (%d, %d): %6.4f" % (remaining_budget, opts.budget, p_r))
    if runif(1, min=0.0, max=1.0)[0] > p_r:
        # return the top ranked item
        candidates = get_first_vals_not_marked(ordered_idxs, hf, n=1, start=1)
    else:
        # potential candidates
        candidates = get_first_vals_not_marked(ordered_idxs, hf,
                                               n=opts.query_search_candidates, start=1)
    return candidates


# a = matrix(rnorm(100, 0, 1), ncol=1)
# oi = order(a, decreasing = T)
# plot(1:100, a[oi])
# p_y_given_x = fit_logistic(a, oi, 0.10)
# lr = glm(c(rep(1, 50), rep(0, 100-50)) ~ a[oi], family="binomial")
# summary(lr)
# coef(lr)
# predict(lr, a[oi], type="response")
# glmnet(a[oi], c(rep(1, 3), rep(0, 100-3)), family=c("binomial"), alpha=0)
def p_logistic(x, ordered_indexes, tau):
    n = nrow(x)
    n1 = int(round(tau * n, 0))
    lbls = np.array(append(list(rep(1, n1, dtype=int)), list(rep(0, n - n1, dtype=int))))
    lr = LogisticRegressionClassifier.fit(x[ordered_indexes, :], lbls)
    p = lr.predict_prob_for_class(x, 1)
    # cls = lr.predict(x, type="class")
    # logger.debug(cls)
    #lr = glmnet(x[ordered_indexes, ], c(rep(1, n1), rep(0, n - n1)), family=c("binomial"), alpha=0)
    #p = predict(lr, x, type="response", s=c(0.01))
    return p


# a = rnorm(100, 0, 100)
# oi = order(a, decreasing = T)
# p_norm(a[oi], oi, 0.05)
# plot(1:100, a[oi])
def p_norm(vals, ordered_indexes, tau):
    n = len(vals)
    n1 = np.round(tau * n, 0)
    s = vals[(n1+1):n]
    p = pnorm(vals, mean=np.mean(s), sd=np.std(s))
    return p


# [exploration_probability(remaining_budget=i, max_budget=3, lam=6.0) for i in range(0,4)]
def exploration_probability(remaining_budget, max_budget, lam=6.0):
    p = np.exp(-lam*(max_budget - remaining_budget)/max_budget)
    return p

