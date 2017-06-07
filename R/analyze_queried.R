rm(list=ls())
library(extrafont)

#plot_subdir = "aad"
plot_subdir = "if_aad"
#plot_subdir = "ha1_hn1_xtau_obv"
#plot_subdir = "noml_only"
#plot_subdir = "no_anomalies"
datapath = "datasets/anomaly"
resultspath = "datasets/results"
if (plot_subdir == "if_aad") {
  resulttype = "if_aad_trees100_samples256_nscore3_tau0.03_xtau_s0.5"
} else {
  resulttype = sprintf("aad_sensitivity_tau0.03%s", ifelse(plot_subdir=="aad", "", paste("_", plot_subdir, sep="")))
}
datasets = c("abalone", "ann_thyroid_1v3", 
             "covtype_sub", 
             "cardiotocography_1", "kddcup_sub",
             "shuttle_sub", "mammography_sub", "yeast")
#datasets = c("cardiotocography_1", "ann_thyroid_1v3")
datasets = c("ann_thyroid_1v3")
#datasets = c("abalone")
#datasets = c("kddcup_sub")
#datasets = c("toy")
#dataset = datasets[1]

resfilename <- function(baseline=F, xtau=F, order_by_violated=F, iforest=F) {
  if (iforest) {
    if (baseline) {
      filename = sprintf("%s-iforest_tau_instance-trees100_samples256_nscore3-top-active-unifprior-Ca100-1_1-fid1-runidx10-bd60-tau0_030-topK0-pseudoanom_always_False-optim_scipy-queried-baseline.csv", dataset)
    } else {
      filename = sprintf("%s-iforest_tau_instance-trees100_samples256_nscore3-top-active-unifprior-Ca100-1_1-fid1-runidx10-bd60-tau0_030-topK0-pseudoanom_always_False-optim_scipy-queried.csv", dataset)
    }
  } else {
    obvsig = ""
    if (order_by_violated) obvsig = "-by_violated"
    constype = "_pairwise"
    if (xtau) constype = "_tau_instance"
    if (baseline) {
      filename = sprintf("%s-aad%s-top-active-unifprior-Ca100-1_1-fid0-runidx0-bd60-tau0_030-topK0-pseudoanom_always_False-optim_scipy%s-queried-baseline.csv", dataset, constype, obvsig)
    } else {
      filename = sprintf("%s-aad%s-top-active-unifprior-Ca100-1_1-fid0-runidx0-bd60-tau0_030-topK0-pseudoanom_always_False-optim_scipy%s-queried.csv", dataset, constype, obvsig)
    }
  }
}

get_best_run <- function(dataset, resultspath, resulttype, orig_labels, xtau=F, order_by_violated=F, iforest=F) {
  # best run is only on the basis of feedback run, not baseline
  queriedfilename = resfilename(baseline=F, xtau=xtau, order_by_violated=order_by_violated, iforest=iforest)
  queriedfile = file.path(resultspath, dataset, resulttype, queriedfilename)
  queried = as.matrix(read.csv(queriedfile, header=F))
  nfound = c()
  for (i in 1:nrow(queried)) {
    qlbls = orig_labels[queried[i, 3:ncol(queried)]]
    found = ifelse(qlbls == "anomaly", 1, 0)
    nfound = c(nfound, sum(found))
  }
  best_run = which(nfound == max(nfound))[1]
}

baseline = F
xtau = F
order_by_violated = F
iforest = plot_subdir == "if_aad"
for (dataset in datasets) {
  origfilename = sprintf("%s_1.csv", dataset)
  lowdimfilename = sprintf("%s_1_tsne.csv", dataset)
  queriedfilename = resfilename(baseline, xtau=xtau, order_by_violated=order_by_violated, iforest=iforest)
  origfile = file.path(datapath, dataset, "fullsamples", origfilename)
  lowdimfile = file.path(datapath, dataset, "fullsamples", lowdimfilename)
  tdata = read.csv(origfile, header=T)
  lowdim = read.csv(lowdimfile, header=T, sep=" ")
  queriedfile = file.path(resultspath, dataset, resulttype, queriedfilename)
  queried = as.matrix(read.csv(queriedfile, header=F))
  
  if (F) {
    queriedfile_b = file.path(resultspath, dataset, resulttype, queriedfilename_b)
    queried_b = as.matrix(read.csv(queriedfile_b, header=F))
    qlbls = tdata$label[queried[2, 3:ncol(queried)]]
    found = ifelse(qlbls == "anomaly", 1, 0)
    found = cumsum(found)
  }
  
  only_best_run = T
  if (only_best_run) {
    best_run = get_best_run(dataset, resultspath, resulttype, orig_labels=tdata$label,
                            xtau=xtau, order_by_violated=order_by_violated, iforest=iforest)
    print (sprintf("%s best_run %d", dataset, best_run))
    qlist = matrix(queried[best_run, 3:ncol(queried)], nrow=1)[1,]
  } else {
    qlist = matrix(queried[, 3:ncol(queried)], nrow=1)[1,]
  }
  qcounts = as.data.frame(table(qlist))
  qcounts$qlist = as.numeric(as.character(qcounts$qlist))
  qcounts$area = qcounts$Freq / max(qcounts$Freq)
  qcounts$area = sqrt(qcounts$area / pi)
  
  n_indexes = setdiff(which(tdata$label=="nominal"), qcounts$qlist)
  a_indexes = setdiff(which(tdata$label=="anomaly"), qcounts$qlist)
  lowdim_noml = as.matrix(lowdim[n_indexes,])
  lowdim_anom = as.matrix(lowdim[a_indexes,])
  queried_points = as.matrix(lowdim[qcounts$qlist,])
  queried_point_labels = tdata$label[qcounts$qlist]
  
  # print (length(n_indexes) + length(a_indexes) + nrow(qcounts))
  
  #plot(0, typ="n", xlim=c(min(lowdim$x), max(lowdim$x)), 
  #     ylim=c(min(lowdim$y), max(lowdim$y)), xlab="x", ylab="y")
  plotdir = file.path("datasets/plots-tmp", "tsne", sprintf("tsne_%s", plot_subdir))
  dir.create(plotdir, recursive=T, showWarnings = F)
  if (baseline) {
    fout = file.path(plotdir, sprintf("num_seen-%s_baseline.pdf", dataset))
    fout_tmp = file.path(plotdir, sprintf("tmp_num_seen-%s_baseline.pdf", dataset))
  } else {
    fout = file.path(plotdir, sprintf("num_seen-%s.pdf", dataset))
    fout_tmp = file.path(plotdir, sprintf("tmp_num_seen-%s.pdf", dataset))
  }
  pdf(file=fout_tmp, family="Arial")
  if (F) {
    symbols(lowdim[qcounts$qlist,1], lowdim[qcounts$qlist,2], 
            fg="pink", bg="pink", circles=qcounts$Freq, 
            xlim=c(min(lowdim$x), max(lowdim$x)), 
            ylim=c(min(lowdim$y), max(lowdim$y)), xlab="x", ylab="y", inches=F)
    points(lowdim_noml[,1], lowdim_noml[,2], col="grey60")
    q_anoms = queried_points[queried_point_labels == "anomaly",]
    q_nomls = queried_points[queried_point_labels == "nominal",]
    #points(queried_points[,1], queried_points[,2], 
    #       col=ifelse(queried_point_labels == "anomaly", "red", "grey60"), pch="+", cex=3.0)
    points(q_nomls[,1], q_nomls[,2], col="grey60", lwd=2, typ="p", pch='+', cex=3.0)
    points(q_anoms[,1], q_anoms[,2], col="red", lwd=2, typ="p", pch='+', cex=3.0)
    points(lowdim_anom[,1], lowdim_anom[,2], col="red", cex=2.5, lwd=2)
  } else if (T) {
    # use this for plotting the results from best run
    q_anoms = queried_points[queried_point_labels == "anomaly",]
    q_nomls = queried_points[queried_point_labels == "nominal",]
    
    plot(lowdim_anom[,1], lowdim_anom[,2], col="red", lwd=2, typ="p", 
         xlim=c(min(lowdim$x), max(lowdim$x)), 
         ylim=c(min(lowdim$y), max(lowdim$y)),
         xlab="x", ylab="y")
    points(lowdim_noml[,1], lowdim_noml[,2], col="grey60", pch='o', cex=1.0)
    points(q_nomls[,1], q_nomls[,2], col="green", lwd=2, typ="p", pch='o', cex=3.0)
    points(lowdim_anom[,1], lowdim_anom[,2], col="blue", cex=2.5, lwd=2, pch='+')
    points(q_anoms[,1], q_anoms[,2], col="red", lwd=2, typ="p", pch='+', cex=3.0)
  } else {
    plot(lowdim_anom[,1], lowdim_anom[,2], col="red", cex=2.5, lwd=2, typ="p", 
         xlim=c(min(lowdim$x), max(lowdim$x)), 
         ylim=c(min(lowdim$y), max(lowdim$y)),
         xlab="x", ylab="y")
    q_anoms = queried_points[queried_point_labels == "anomaly",]
    points(q_anoms[,1], q_anoms[,2], col="red", cex=2.5, lwd=2, typ="p", pch='+')
    #points(queried_points[,1], queried_points[,2],
    #       col=ifelse(queried_point_labels == "anomaly", "red", "grey60"), cex=2.5, lwd=2, typ="p")
  }
  dev.off()
  embed_fonts(fout_tmp, outfile=fout)
}

if (F) {
  source("./R/analyze_queried.R")
}
