from loda_support import *
from query_search import *


class Query(object):
    def __init__(self, opts=None, **kwargs):
        self.opts = opts

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
        lbls = kwargs.get("lbls")
        w = kwargs.get("w")
        hf = kwargs.get("hf")
        remaining_budget = kwargs.get("remaining_budget")

        a = None
        y = None
        k = self.opts.query_search_depth

        best_query_and_value = get_next_query_and_utility(x=x, lbls=lbls,
                                                          w=w, hf=hf,
                                                          remaining_budget=remaining_budget,
                                                          k=k, a=a, y=y, opts=self.opts)
        return best_query_and_value.action
