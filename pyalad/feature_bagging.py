"""
This code merely aggregates precomputed scores from an ensemble.

Related publication:
    Feature Bagging for Outlier Detection by Aleksandar Lazarevic and Vipin Kumar, KDD 2005.
"""

from alad_support import *


def aggregate_scores_breadth_first(opts):

    scores = load_ensemble_scores(opts.scoresfile, header=True)

    n, m = scores.fmat.shape

    score_ranks = np.zeros(shape=(n, m), dtype=int)
    for i in range(m):
        score_ranks[:, i] = order(scores.fmat[:, i], decreasing=True)

    # logger.debug(list(scores.fmat[score_ranks[:, 0], 0]))

    bt = get_budget_topK(n, opts)
    final_ranks = np.zeros(nrow(scores.fmat), dtype=int) - 1
    fp = 0
    for i in range(n):
        insts_at_rank_i = score_ranks[i, :]
        for j in insts_at_rank_i:
            pos = np.where(final_ranks == j)[0]
            if len(pos) == 0:
                final_ranks[fp] = j
                fp += 1
        if fp >= n:
            break

    queried = list(final_ranks[0:bt.budget])
    # the first two positions are (fid,runidx) = (0,0)
    num_seen = np.zeros(len(queried) + 2)
    num_seen[2:len(num_seen)] = np.cumsum(scores.lbls[queried])
    num_seen = num_seen.reshape((1, len(num_seen)))
    logger.debug(num_seen.shape)

    return num_seen

