import os
import numpy as np
import matplotlib.pyplot as plt

import logging
from pandas import DataFrame

from app_globals import *
from alad_support import *
from r_support import matrix, cbind

from alad_iforest import *
from data_plotter import *
from results_support import *


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


def iforest_unit_tests_battery(X_train, labels, mdl, metrics, opts,
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

    output_iforest_original = False
    output_transformed_to_file = False
    test_loss_grad = False
    plot_dataset = data_2D and True
    plot_rectangular_regions = plot_dataset and False
    plot_iforest_contours = data_2D and False
    plot_baseline = data_2D and False
    plot_aad = metrics is not None and data_2D and False

    pdfpath_baseline = "%s/if_baseline.pdf" % outputdir
    pdfpath_orig_if_contours = "%s/if_contours.pdf" % outputdir

    logger.debug("Number of regions: %d" % len(mdl.d))
    X_train_new = mdl.transform_to_region_features(X_train, dense=False)

    if plot_dataset:
        plot_dataset_2D(X_train, labels, mdl, plot_rectangular_regions, regcols, outputdir)

    if output_iforest_original:
        n_found = evaluate_iforest_original(X_train, labels, opts.budget, mdl, x_new=X_train_new)
        np.savetxt(os.path.join(outputdir, "iforest_original_num_found_%s.csv" % dataset_name),
                   n_found, fmt='%3.2f', delimiter=",")

    if plot_iforest_contours:
        plot_iforest_contours_2D(X_train, labels, xx, yy, opts.budget, mdl,
                                 pdfpath_orig_if_contours, dash_xy, dash_wh)

    if output_transformed_to_file:
        write_sparsemat_to_file(os.path.join(outputdir, "iforest_features.csv"),
                                X_train_new, fmt='%3.2f', delimiter=",")
        x_tmp = np.vstack((mdl.d, mdl.node_samples, mdl.frac_insts))
        write_sparsemat_to_file(os.path.join(outputdir, "iforest_node_info.csv"),
                                x_tmp.T, fmt='%3.2f', delimiter=",")

    if test_loss_grad:
        test_iforest_loss_grad(X_train_new, labels, mdl, opts)

    if plot_baseline:
        plot_iforest_baseline_contours_2D(X_train, labels, X_train_new, xx, yy, opts.budget, mdl,
                                          pdfpath_baseline, dash_xy, dash_wh)

    if plot_aad and metrics is not None:
        plot_aad_2D(X_train, labels, X_train_new, xx, yy, mdl,
                    metrics, outputdir, dash_xy, dash_wh)


def get_original_iforest_results(x, y, if_model, fid, runidx, opts):
    num_seen = np.zeros(shape=(1, opts.budget + 2), dtype=int)
    counts = evaluate_iforest_original(x, y, opts.budget, if_model)
    num_seen[0, 0:2] = [fid, runidx]
    num_seen[0, 2:(opts.budget + 2)] = counts
    return num_seen


def plot_aad_2D(x, y, x_iforest, xx, yy, if_model, metrics,
                outputdir, dash_xy, dash_wh):
    # use this to plot the AAD feedback

    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_if = if_model.transform_to_region_features(x_test, dense=False)

    queried = np.array(metrics.queried)
    for i, q in enumerate(queried):
        pdfpath = "%s/iter_%02d.pdf" % (outputdir, i)
        dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
        pl = dp.get_next_plot()

        w = metrics.all_weights[i, :]
        Z = if_model.get_score(x_if, w)
        Z = Z.reshape(xx.shape)

        pl.contourf(xx, yy, Z, 20)

        dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"}, s=25)
        # print queried[np.arange(i+1)]
        # print X_train[queried[np.arange(i+1)], :]
        dp.plot_points(matrix(x[queried[np.arange(i+1)], :], nrow=i+1),
                       pl, labels=y[queried[np.arange(i+1)]], defaultcol="red",
                       lbl_color_map={0: "green", 1: "red"}, edgecolor="black",
                       marker=matplotlib.markers.MarkerStyle('o', fillstyle=None), s=35)

        # plot the sidebar
        anom_scores = if_model.get_score(x_iforest, w)
        anom_order = np.argsort(-anom_scores)
        anom_idxs = np.where(y[anom_order] == 1)[0]
        dash = 1 - (anom_idxs * 1.0 / x.shape[0])
        plot_sidebar(dash, dash_xy, dash_wh, pl)

        dp.close()


def test_iforest_loss_grad(x_iforest, y, if_model, opts):
    n = x_iforest.shape[0]
    bt = get_budget_topK(n, opts)

    w = np.ones(len(if_model.d), dtype=float)
    w = w / w.dot(w)  # normalized uniform weights

    qval = if_model.get_aatp_quantile(x_iforest, w, bt.topK)
    w_unifprior = np.ones(len(if_model.d), dtype=float)
    w_unifprior = w_unifprior / w_unifprior.dot(w_unifprior)
    print "topK=%d, budget=%d, qval=%8.5f" % (bt.topK, bt.budget, qval)
    theta = np.zeros(w.shape, dtype=float)
    loss = if_model.if_aad_loss_linear(w, x_iforest[1:10, :], y[1:10], qval,
                                       Ca=opts.Ca, Cn=opts.Cn,
                                       withprior=opts.withprior, w_prior=w_unifprior,
                                       sigma2=opts.priorsigma2)
    print "loss: %f" % loss
    loss_grad = if_model.if_aad_loss_gradient_linear(w, x_iforest[1:10, :], y[1:10], qval,
                                                     Ca=opts.Ca, Cn=opts.Cn,
                                                     withprior=opts.withprior, w_prior=w_unifprior,
                                                     sigma2=opts.priorsigma2)
    logger.debug("loss_grad")
    logger.debug(loss_grad)


def evaluate_iforest_original(x, y, budget, if_model, x_new=None):
    original_scores = 0.5 - if_model.decision_function(x)
    queried = np.argsort(-original_scores)

    n_found_orig = np.cumsum(y[queried[np.arange(budget)]])
    # logger.debug("original isolation forest:")
    # logger.debug(n_found_orig)

    if x_new is not None:
        w = np.ones(len(if_model.d), dtype=float)
        w = w / w.dot(w)  # normalized uniform weights
        agg_scores = if_model.get_score(x_new, w)
        queried = np.argsort(-agg_scores)
        n_found_baseline = np.cumsum(y[queried[np.arange(budget)]])
        n_found = np.vstack((n_found_baseline, n_found_orig)).T
    else:
        n_found = n_found_orig.T
    return n_found


def plot_iforest_baseline_contours_2D(x, y, x_iforest, xx, yy, budget, if_model,
                                      pdfpath_if_contours, dash_xy, dash_wh):
    # use this to plot baseline query points.

    w = np.ones(len(if_model.d), dtype=float)
    w = w / w.dot(w)  # normalized uniform weights

    baseline_scores = if_model.get_score(x_iforest, w)
    queried = np.argsort(-baseline_scores)

    n_found = np.cumsum(y[queried[np.arange(budget)]])
    print n_found

    dp = DataPlotter(pdfpath=pdfpath_if_contours, rows=1, cols=1)
    pl = dp.get_next_plot()

    x_test = np.c_[xx.ravel(), yy.ravel()]
    x_if = if_model.transform_to_region_features(x_test, dense=False)
    y_if = if_model.get_score(x_if, w)
    Z = y_if.reshape(xx.shape)

    pl.contourf(xx, yy, Z, 20)

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


def plot_iforest_contours_2D(x, y, xx, yy, budget, if_model, pdfpath_if_contours, dash_xy, dash_wh):
    # Original IsolationForest contours
    baseline_scores = 0.5 - if_model.decision_function(x)
    queried = np.argsort(-baseline_scores)

    n_found = np.cumsum(y[queried[np.arange(budget)]])
    print n_found

    Z_if = 0.5 - if_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_if = Z_if.reshape(xx.shape)

    dp = DataPlotter(pdfpath=pdfpath_if_contours, rows=1, cols=1)
    pl = dp.get_next_plot()
    pl.contourf(xx, yy, Z_if, 20)

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


def plot_dataset_2D(x, y, if_model, plot_regions, regcols, pdf_folder):
    # use this to plot the dataset

    treesig = "_%d_trees" % if_model.n_estimators if plot_regions else ""
    pdfpath_dataset = "%s/synth_dataset%s.pdf" % (pdf_folder, treesig)
    dp = DataPlotter(pdfpath=pdfpath_dataset, rows=1, cols=1)
    pl = dp.get_next_plot()

    # dp.plot_points(x, pl, labels=y, lbl_color_map={0: "grey", 1: "red"})
    dp.plot_points(x[y==0, :], pl, labels=y[y==0], defaultcol="grey")
    dp.plot_points(x[y==1, :], pl, labels=y[y==1], defaultcol="red", s=26, linewidths=1.5)

    if plot_regions:
        # plot the isolation forest tree regions
        axis_lims = (plt.xlim(), plt.ylim())
        for i, regions in enumerate(if_model.regions_in_forest):
            for region in regions:
                region = region.region
                plot_rect_region(pl, region, regcols[i % len(regcols)], axis_lims)
    dp.close()

