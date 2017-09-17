import os
import numpy as np
import matplotlib.pyplot as plt

import logging
from pandas import DataFrame

from app_globals import *
from alad_support import *
from r_support import matrix, cbind

from forest_aad_detector import *
from data_plotter import *
from results_support import *
from gp_support import *


def get_queried_indexes(scores, labels, opts):
    # logger.debug("computing queried indexes...")
    queried = np.argsort(-scores)[0:opts.budget]
    num_seen = np.cumsum(labels[queried[np.arange(opts.budget)]])
    return num_seen, queried


def write_baseline_query_indexes(queried_info, opts):
    logger.debug("writing baseline queries...")
    queried = np.zeros(shape=(len(queried_info), opts.budget + 2), dtype=int)
    num_seen = np.zeros(shape=(len(queried_info), opts.budget + 2), dtype=int)
    for i, info in enumerate(queried_info):
        num_seen[i, 2:(opts.budget + 2)] = info[0]
        num_seen[i, 0] = 1
        queried[i, 2:(opts.budget + 2)] = info[1] + 1  # make indexes relative 1, *not* 0
        queried[i, 0] = 1
    prefix = opts.get_alad_metrics_name_prefix()
    baseline_file = os.path.join(opts.resultsdir, "%s-baseline.csv" % (prefix,))
    # np.savetxt(baseline_file, num_seen, fmt='%d', delimiter=',')
    queried_idxs_baseline_file = os.path.join(opts.resultsdir, "%s-queried-baseline.csv" % (prefix,))
    np.savetxt(queried_idxs_baseline_file, queried, fmt='%d', delimiter=',')


def forest_aad_unit_tests_battery(X_train, labels, model, metrics, opts,
                                  outputdir, dataset_name=""):

    data_2D = X_train.shape[1] == 2

    regcols = ["red", "blue", "green", "brown", "cyan", "pink", "orange", "magenta", "yellow", "violet"]

    xx = None; yy = None
    if data_2D:
        # plot the line, the samples, and the nearest vectors to the plane
        xx, yy = np.meshgrid(np.linspace(-4, 8, 50), np.linspace(-4, 8, 50))

    # sidebar coordinates and dimensions for showing rank locations of true anomalies
    dash_xy = (-4.0, -2.0)  # bottom-left (x,y) coordinates
    dash_wh = (0.4, 8)  # width, height

    output_forest_original = False
    output_transformed_to_file = False
    test_loss_grad = False
    plot_dataset = data_2D and True
    plot_rectangular_regions = plot_dataset and True
    plot_forest_contours = data_2D and True
    plot_baseline = data_2D and False
    plot_aad = metrics is not None and data_2D and True

    pdfpath_baseline = "%s/tree_baseline.pdf" % outputdir
    pdfpath_orig_if_contours = "%s/score_contours.pdf" % outputdir

    logger.debug("Number of regions: %d" % len(model.d))

    tm = Timer()
    X_train_new = model.transform_to_region_features(X_train, dense=False)
    logger.debug(tm.message("transformed input to region features"))

    if plot_dataset:
        tm.start()
        plot_dataset_2D(X_train, labels, model, plot_rectangular_regions, regcols, outputdir)
        logger.debug(tm.message("plotted dataset"))

    if output_forest_original:
        n_found = evaluate_forest_original(X_train, labels, opts.budget, model, x_new=X_train_new)
        np.savetxt(os.path.join(outputdir, "iforest_original_num_found_%s.csv" % dataset_name),
                   n_found, fmt='%3.2f', delimiter=",")

    if plot_forest_contours:
        tm.start()
        plot_forest_contours_2D(X_train, labels, xx, yy, opts.budget, model,
                                pdfpath_orig_if_contours, dash_xy, dash_wh)
        logger.debug(tm.message("plotted contours"))

    if output_transformed_to_file:
        write_sparsemat_to_file(os.path.join(outputdir, "forest_features.csv"),
                                X_train_new, fmt='%3.2f', delimiter=",")
        x_tmp = np.vstack((model.d, model.node_samples, model.frac_insts))
        write_sparsemat_to_file(os.path.join(outputdir, "forest_node_info.csv"),
                                x_tmp.T, fmt='%3.2f', delimiter=",")

    if test_loss_grad:
        test_forest_loss_grad(X_train_new, labels, model, opts)

    if plot_baseline:
        plot_forest_baseline_contours_2D(X_train, labels, X_train_new, xx, yy, opts.budget, model,
                                         pdfpath_baseline, dash_xy, dash_wh)

    if plot_aad and metrics is not None:
        plot_aad_2D(X_train, labels, X_train_new, xx, yy, model,
                    metrics, outputdir, dash_xy, dash_wh)

        if False:
            plot_aad_gp(X_train, labels, X_train_new, xx, yy, model,
                        metrics, outputdir, dash_xy, dash_wh)

        if False:
            plot_aad_score_var(X_train, labels, X_train_new, xx, yy, model,
                               metrics, outputdir, dash_xy, dash_wh)


def plot_aad_2D(x, y, x_forest, xx, yy, forest, metrics,
                outputdir, dash_xy, dash_wh):
    # use this to plot the AAD feedback

    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_if = forest.transform_to_region_features(x_test, dense=False)

    queried = np.array(metrics.queried)
    for i, q in enumerate(queried):
        pdfpath = "%s/iter_%02d.pdf" % (outputdir, i)
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()

        w = metrics.all_weights[i, :]
        Z = forest.get_score(x_if, w)
        Z = Z.reshape(xx.shape)

        pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))

        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        # print queried[np.arange(i+1)]
        # print X_train[queried[np.arange(i+1)], :]
        dp.plot_points(matrix(x[queried[np.arange(i+1)], :], nrow=i+1),
                       pl, labels=y[queried[np.arange(i+1)]], defaultcol="red",
                       lbl_color_map={0: "green", 1: "red"}, edgecolor=None, facecolors=True,
                       marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)

        # plot the sidebar
        anom_scores = forest.get_score(x_forest, w)
        anom_order = np.argsort(-anom_scores)
        anom_idxs = np.where(y[anom_order] == 1)[0]
        dash = 1 - (anom_idxs * 1.0 / x.shape[0])
        plot_sidebar(dash, dash_xy, dash_wh, pl)

        dp.close()


def plot_aad_score_var(x, y, x_forest, xx, yy, forest, metrics,
                outputdir, dash_xy, dash_wh):
    # use this to plot the AAD feedback

    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_test_forest = forest.transform_to_region_features(x_test, dense=False)

    queried = np.array(metrics.queried)
    for i, q in enumerate(queried):
        pdfpath = "%s/score_iter_%02d.pdf" % (outputdir, i)
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()

        w = metrics.all_weights[i, :]
        s_train = forest.get_score(x_forest, w)
        ranked_indexes = np.argsort(-s_train, )
        # s_test = forest.get_score(x_test_forest, w)
        test_indexes = metrics.test_indexes[i]
        score_eval_set = x_test_forest
        score_var, test_indexes, v_eval = \
            get_score_variances(x=x_forest, w=w,
                                ordered_indexes=ranked_indexes,
                                queried_indexes=queried,
                                n_test=len(test_indexes) if test_indexes is not None else 10,
                                test_indexes=test_indexes,
                                eval_set=score_eval_set,
                                n_closest=9)
        qpos = np.argmax(score_var)
        q = test_indexes[qpos]
        logger.debug("score_var:\n%s\ntest_indexes:\n%s" %
                     (str(list(score_var)), str(list(test_indexes))))
        logger.debug("qpos: %d, query instance: %d, var: %f, queried:%s" %
                     (qpos, q, score_var[qpos], str(list(queried[np.arange(i)]))))

        if score_eval_set is not None:
            Z = v_eval.reshape(xx.shape)

            levels = np.linspace(np.min(v_eval), np.max(v_eval), 20)
            CS = pl.contourf(xx, yy, Z, levels, cmap=plt.cm.get_cmap('jet'))
            cbar = plt.colorbar(CS)
            cbar.ax.set_ylabel('score variance')

        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        dp.plot_points(x[test_indexes, :], pl, marker='o', defaultcol='magenta',
                       s=60, edgecolor='magenta', facecolors='none')
        dp.plot_points(matrix(x[queried[np.arange(i+1)], :], nrow=i+1),
                       pl, labels=y[queried[np.arange(i+1)]], defaultcol="red",
                       lbl_color_map={0: "green", 1: "red"}, edgecolor=None, facecolors=True,
                       marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)

        # plot the sidebar
        anom_scores = forest.get_score(x_forest, w)
        anom_order = np.argsort(-anom_scores)
        anom_idxs = np.where(y[anom_order] == 1)[0]
        dash = 1 - (anom_idxs * 1.0 / x.shape[0])
        plot_sidebar(dash, dash_xy, dash_wh, pl)

        dp.close()


def plot_aad_gp(x, y, x_forest, xx, yy, forest, metrics,
                outputdir, dash_xy, dash_wh):
    # use this to plot the AAD feedback

    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_test_forest = forest.transform_to_region_features(x_test, dense=False)

    queried = np.array(metrics.queried)
    for i, q in enumerate(queried):
        pdfpath = "%s/gp_iter_%02d.pdf" % (outputdir, i)
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()

        w = metrics.all_weights[i, :]
        s_train = forest.get_score(x_forest, w)
        ranked_indexes = np.argsort(-s_train, )
        # s_test = forest.get_score(x_test_forest, w)

        gp_eval_set = x_test_forest
        gp_score, gp_var, train_indexes, test_indexes, v_eval = \
            get_gp_predictions(x=x_forest, y=s_train,
                               orig_x=x,
                               ordered_indexes=ranked_indexes,
                               queried_indexes=queried,
                               n_train=100, n_test=30, length_scale=40,
                               eval_set=gp_eval_set, orig_eval_set=x_test,
                               n_closest=9)
        logger.debug("gp_var:\n%s\ntest_indexes:\n%s" % (str(list(gp_var)), str(list(test_indexes))))

        if gp_eval_set is not None:
            Z = v_eval.reshape(xx.shape)

            levels = np.linspace(0., 1., 20)
            CS = pl.contourf(xx, yy, Z, levels, cmap=plt.cm.get_cmap('jet'))
            cbar = plt.colorbar(CS)
            cbar.ax.set_ylabel('score variance')

        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        dp.plot_points(x[train_indexes, :], pl, marker='o', defaultcol='blue',
                       s=35, edgecolor='blue', facecolors='none')
        dp.plot_points(x[test_indexes, :], pl, marker='o', defaultcol='magenta',
                       s=60, edgecolor='magenta', facecolors='none')
        # print queried[np.arange(i+1)]
        # print X_train[queried[np.arange(i+1)], :]
        dp.plot_points(matrix(x[queried[np.arange(i+1)], :], nrow=i+1),
                       pl, labels=y[queried[np.arange(i+1)]], defaultcol="red",
                       lbl_color_map={0: "green", 1: "red"}, edgecolor="black",
                       marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)

        # plot the sidebar
        anom_scores = forest.get_score(x_forest, w)
        anom_order = np.argsort(-anom_scores)
        anom_idxs = np.where(y[anom_order] == 1)[0]
        dash = 1 - (anom_idxs * 1.0 / x.shape[0])
        plot_sidebar(dash, dash_xy, dash_wh, pl)

        dp.close()


def test_forest_loss_grad(x_forest, y, model, opts):
    n = x_forest.shape[0]
    bt = get_budget_topK(n, opts)

    w = np.ones(len(model.d), dtype=float)
    w = w / w.dot(w)  # normalized uniform weights

    qval = model.get_aatp_quantile(x_forest, w, bt.topK)
    w_unifprior = np.ones(len(model.d), dtype=float)
    w_unifprior = w_unifprior / w_unifprior.dot(w_unifprior)
    print "topK=%d, budget=%d, qval=%8.5f" % (bt.topK, bt.budget, qval)
    theta = np.zeros(w.shape, dtype=float)
    loss = model.if_aad_loss_linear(w, x_forest[1:10, :], y[1:10], qval,
                                    Ca=opts.Ca, Cn=opts.Cn,
                                    withprior=opts.withprior, w_prior=w_unifprior,
                                    sigma2=opts.priorsigma2)
    print "loss: %f" % loss
    loss_grad = model.if_aad_loss_gradient_linear(w, x_forest[1:10, :], y[1:10], qval,
                                                  Ca=opts.Ca, Cn=opts.Cn,
                                                  withprior=opts.withprior, w_prior=w_unifprior,
                                                  sigma2=opts.priorsigma2)
    logger.debug("loss_grad")
    logger.debug(loss_grad)


def evaluate_forest_original(x, y, budget, forest, x_new=None):
    original_scores = 0.5 - forest.decision_function(x)
    queried = np.argsort(-original_scores)

    n_found_orig = np.cumsum(y[queried[np.arange(budget)]])
    # logger.debug("original isolation forest:")
    # logger.debug(n_found_orig)

    if x_new is not None:
        w = np.ones(len(forest.d), dtype=float)
        w = w / w.dot(w)  # normalized uniform weights
        agg_scores = forest.get_score(x_new, w)
        queried = np.argsort(-agg_scores)
        n_found_baseline = np.cumsum(y[queried[np.arange(budget)]])
        n_found = np.vstack((n_found_baseline, n_found_orig)).T
    else:
        n_found = np.reshape(n_found_orig, (1, len(n_found_orig)))
    return n_found


def plot_forest_baseline_contours_2D(x, y, x_forest, xx, yy, budget, forest,
                                     pdfpath_contours, dash_xy, dash_wh):
    # use this to plot baseline query points.

    w = np.ones(len(forest.d), dtype=float)
    w = w / w.dot(w)  # normalized uniform weights

    baseline_scores = forest.get_score(x_forest, w)
    queried = np.argsort(-baseline_scores)

    n_found = np.cumsum(y[queried[np.arange(budget)]])
    print n_found

    dp = DataPlotter(pdfpath=pdfpath_contours, rows=1, cols=1)
    pl = dp.get_next_plot()

    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_if = forest.transform_to_region_features(x_test, dense=False)
    y_if = forest.get_score(x_if, w)
    Z = y_if.reshape(xx.shape)

    pl.contourf(xx, yy, Z, 20, cmap=plt.cm.get_cmap('jet'))

    dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
    # print queried[np.arange(i+1)]
    # print X_train[queried[np.arange(i+1)], :]
    dp.plot_points(matrix(x[queried[np.arange(budget)], :], nrow=budget),
                   pl, labels=y[queried[np.arange(budget)]], defaultcol="red",
                   lbl_color_map={0: "green", 1: "red"}, edgecolor="black",
                   marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)

    # plot the sidebar
    anom_idxs = np.where(y[queried] == 1)[0]
    dash = 1 - (anom_idxs * 1.0 / x.shape[0])
    plot_sidebar(dash, dash_xy, dash_wh, pl)

    dp.close()


def plot_forest_contours_2D(x, y, xx, yy, budget, forest, pdfpath_contours, dash_xy, dash_wh):
    # Original detector contours
    baseline_scores = 0.5 - forest.decision_function(x)
    queried = np.argsort(-baseline_scores)
    # logger.debug("baseline scores:%s\n%s" % (str(baseline_scores.shape), str(list(baseline_scores))))

    n_found = np.cumsum(y[queried[np.arange(budget)]])
    print n_found

    Z_if = 0.5 - forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_if = Z_if.reshape(xx.shape)

    dp = DataPlotter(pdfpath=pdfpath_contours, rows=1, cols=1)
    pl = dp.get_next_plot()
    pl.contourf(xx, yy, Z_if, 20, cmap=plt.cm.get_cmap('jet'))

    dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"})

    dp.plot_points(matrix(x[queried[np.arange(budget)], :], nrow=budget),
                   pl, labels=y[queried[np.arange(budget)]], defaultcol="red",
                   lbl_color_map={0: "green", 1: "red"}, edgecolor="black",
                   marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)

    # plot the sidebar
    anom_idxs = np.where(y[queried] == 1)[0]
    dash = 1 - (anom_idxs * 1.0 / x.shape[0])
    plot_sidebar(dash, dash_xy, dash_wh, pl)

    dp.close()


def plot_dataset_2D(x, y, forest, plot_regions, regcols, pdf_folder):
    # use this to plot the dataset

    treesig = "_%d_trees" % forest.n_estimators if plot_regions else ""
    pdfpath_dataset = "%s/synth_dataset%s.pdf" % (pdf_folder, treesig)
    dp = DataPlotter(pdfpath=pdfpath_dataset, rows=1, cols=1)
    pl = dp.get_next_plot()

    # dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"})
    dp.plot_points(x[y==0, :], pl, labels=y[y==0], defaultcol="grey")
    dp.plot_points(x[y==1, :], pl, labels=y[y==1], defaultcol="red", s=26, linewidths=1.5)

    if plot_regions:
        # plot the isolation forest tree regions
        axis_lims = (plt.xlim(), plt.ylim())
        for i, regions in enumerate(forest.regions_in_forest):
            for region in regions:
                region = region.region
                plot_rect_region(pl, region, regcols[i % len(regcols)], axis_lims)
    dp.close()


def prepare_forest_aad_debug_args():
    datasets = ["abalone", "ann_thyroid_1v3", "cardiotocography_1", "covtype_sub",
                "kddcup_sub", "mammography_sub", "shuttle_sub", "yeast", "toy", "toy2"]

    dataset = datasets[9]
    datapath = "./datasets/anomaly/%s/fullsamples/%s_1.csv" % (dataset, dataset)
    outputdir = "./temp"

    budget = 35
    n_runs = 2
    inference_type = AAD_RSFOREST
    # inference_type = AAD_HSTREES
    # inference_type = AAD_IFOREST
    sigma2 = 0.5
    n_jobs = 4
    add_leaves_only = False
    plot2D = True

    streaming = True
    stream_window = 64
    allow_stream_update = True

    if inference_type == AAD_IFOREST:
        n_trees = 100
        forest_max_depth = 100
        score_type = IFOR_SCORE_TYPE_CONST
        ensemble_score = ENSEMBLE_SCORE_LINEAR
        Ca = 100.
        Cx = 0.001
    elif inference_type == AAD_HSTREES:
        n_trees = 25
        forest_max_depth = 7
        score_type = HST_SCORE_TYPE
        ensemble_score = ENSEMBLE_SCORE_LINEAR
        Ca = 1.
        Cx = 0.001
    elif inference_type == AAD_RSFOREST:
        n_trees = 30
        forest_max_depth = 7
        score_type = RSF_LOG_SCORE_TYPE
        # score_type = RSF_SCORE_TYPE
        # score_type = ORIG_TREE_SCORE_TYPE
        ensemble_score = ENSEMBLE_SCORE_LINEAR
        Ca = 1.
        Cx = 0.001
    else:
        raise ValueError("Invalid inference type %s" % inference_type)

    args = get_forest_aad_args(dataset=dataset, n_trees=n_trees,
                               detector_type=inference_type,
                               forest_add_leaf_nodes_only=add_leaves_only,
                               forest_score_type=score_type,
                               forest_max_depth=forest_max_depth,
                               ensemble_score=ensemble_score,
                               sigma2=sigma2, Ca=Ca, Cx=Cx,
                               budget=budget, reruns=n_runs, n_jobs=n_jobs,
                               log_file="./temp/aad.log", plot2D=plot2D,
                               streaming=streaming, stream_window=stream_window,
                               allow_stream_update=allow_stream_update)
    args.datafile = datapath
    args.resultsdir = os.path.join(outputdir, args.dataset,
                                   "%s_trees%d_samples%d_q%d_bd%d_nscore%d%s_tau%1.2f_sig%4.3f_ca%1.0f_cx%4.3f_%s%s%s" %
                                   (detector_types[args.detector_type], args.ifor_n_trees, args.ifor_n_samples,
                                    args.querytype, args.budget, args.forest_score_type,
                                    "" if not args.forest_add_leaf_nodes_only else "_leaf",
                                    args.tau,
                                    args.sigma2, args.Ca, args.Cx, ensemble_score_names[ensemble_score],
                                    "" if args.detector_type == AAD_IFOREST else "_d%d" % args.forest_max_depth,
                                    "_stream" if streaming else ""))
    dir_create(args.resultsdir)
    return args