from loda_support import *
from query_search import *
from gp_support import *

class Query(object):
    def __init__(self, opts=None, **kwargs):
        self.opts = opts
        self.test_indexes = None

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        pass

    @staticmethod
    def get_initial_query_state(querytype, opts, **kwargs):
        if querytype == QUERY_DETERMINISIC:
            return QueryTop(opts=opts, **kwargs)
        elif querytype == QUERY_BETA_ACTIVE:
            raise NotImplementedError("Beta active query strategy not implemented yet")
        elif querytype == QUERY_QUANTILE:
            return QueryQuantile(opts=opts, **kwargs)
        elif querytype == QUERY_RANDOM:
            return QueryRandom(opts=opts, **kwargs)
        elif querytype == QUERY_SEQUENTIAL:
            return QuerySequential(opts=opts, **kwargs)
        elif querytype == QUERY_GP:
            return QueryGP(opts=opts, **kwargs)
        elif querytype == QUERY_SCORE_VAR:
            return QueryScoreVar(opts=opts, **kwargs)
        else:
            raise ValueError("Invalid query type %d" % (querytype,))


class QueryTop(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        ordered_indexes = kwargs.get("ordered_indexes")
        queried_items = kwargs.get("queried_items")
        items = get_first_vals_not_marked(ordered_indexes, queried_items, start=0, n=1)
        if len(items) > 0:
            return items[0]
        return None


class QueryGP(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.budget = kwargs.get("budget")
        self.s = 0
        self.f = 0

    def update_query_state(self, **kwargs):
        rewarded = kwargs.get("rewarded")
        if rewarded:
            self.s += 1
        else:
            self.f += 1

    def get_next_query(self, **kwargs):
        ordered_indexes = kwargs.get("ordered_indexes")
        queried_items = kwargs.get("queried_items")
        r, bmean = bernoulli_explore_exploit_sample(a=self.a, b=self.b, s=self.s, f=self.f,
                                                    budget=self.budget, mean=True)
        logger.debug("get_next_query: r=%d (%s), prob explore=%f" %
                     (r, "exploit" if r == 1 else "explore", bmean))
        if True and r == 1:
            # exploit
            items = get_first_vals_not_marked(ordered_indexes, queried_items, start=0, n=1)
            if len(items) > 0:
                return items[0]
        else:
            # explore
            x = kwargs.get("x")
            anom_score = kwargs.get("anom_score")
            gp_score, gp_var, train_indexes, test_indexes, _ = \
                get_gp_predictions(x, anom_score,
                                   ordered_indexes=ordered_indexes,
                                   queried_indexes=queried_items,
                                   n_train=100, n_test=self.opts.n_explore)
            if False:
                logger.debug("gp_var:\n%s\ntrain_indexes:%s\ntest_indexes:\n%s" %
                             (str(list(gp_var)), str(list(train_indexes)), str(list(test_indexes))))
            qpos = np.argmax(gp_var)
            q = test_indexes[qpos]
            if False: logger.debug("query instance: %d" % q)
            return q
        return None


class QueryScoreVar(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.budget = kwargs.get("budget")

        self.stats = np.zeros(shape=(2, 2), dtype=float)  # 2x2 matrix storing success/failure for explore/explore
        self.explore = False

    def update_query_state(self, **kwargs):
        rewarded = kwargs.get("rewarded")
        if rewarded:
            r = np.array([1., 0.], dtype=float)  # increment success counts
        else:
            r = np.array([0., 1.], dtype=float)  # increment failure counts
        self.stats[1 if self.explore else 0, :] += r
        logger.debug("rewarded: %s, explore: %s, updated reward matrix:\n%s" %
                     (rewarded, self.explore, str(self.stats)))

    def get_next_query(self, **kwargs):
        self.test_indexes = None
        ordered_indexes = kwargs.get("ordered_indexes")
        queried_items = kwargs.get("queried_items")
        action, samples, bmean = thompson_sample(a=self.a, b=self.b,
                                                 reward_history=self.stats, mean=True)
        logger.debug("get_next_query: action=%d (%s), prob exploit/explore=%s" %
                     (action, "explore" if action == 1 else "exploit", str(bmean)))
        self.explore = True if action == 1 else False
        if not self.explore and True:
            # exploit
            items = get_first_vals_not_marked(ordered_indexes, queried_items, start=0, n=1)
            if len(items) > 0:
                return items[0]
        else:
            # explore
            x = kwargs.get("x")
            w = kwargs.get("w")
            vars, test_indexes, _ = \
                get_score_variances(x, w,
                                    ordered_indexes=ordered_indexes,
                                    queried_indexes=queried_items,
                                    n_test=self.opts.n_explore)
            self.test_indexes = test_indexes
            if False:
                logger.debug("score_var:\n%s\ntest_indexes:\n%s" %
                             (str(list(vars)), str(list(test_indexes))))
            qpos = np.argmax(vars)
            q = test_indexes[qpos]
            if False:
                logger.debug("qpos: %d, query instance: %d, var: %f, queried:%s" %
                             (qpos, q, vars[qpos], str(list(queried_items))))
            return q
        return None


class QueryQuantile(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        pass


class QueryRandom(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        maxpos = kwargs.get("maxpos")
        ordered_indexes = kwargs.get("ordered_indexes")
        queried_items = kwargs.get("queried_items")
        q = sample(range(maxpos), 1)
        item = get_first_vals_not_marked(ordered_indexes, queried_items, start=q)
        return item


class QuerySequential(Query):
    def __init__(self, opts=None, **kwargs):
        Query.__init__(self, opts)

    def update_query_state(self, **kwargs):
        pass

    def get_next_query(self, **kwargs):
        x = kwargs.get("x")
        labels = kwargs.get("labels")
        w = kwargs.get("w")
        hf = kwargs.get("hf")
        remaining_budget = kwargs.get("remaining_budget")

        a = None
        y = None
        k = self.opts.query_search_depth

        best_query_and_value = get_next_query_and_utility(x=x, lbls=labels,
                                                          w=w, hf=hf,
                                                          remaining_budget=remaining_budget,
                                                          k=k, a=a, y=y, opts=self.opts)
        return best_query_and_value.action
