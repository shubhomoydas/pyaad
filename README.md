# pyaad
Python version of Active Anomaly Discovery

Python libraries required:
--------------------------
numpy
scipy
scikit-learn
cvxopt
pandas
ranking
statsmodels
matplotlib


This codebase has four different algorithms:
  - The LODA based AAD
  - The Isolation Forest based AAD
  - HS Trees based AAD (with streaming support **still needs better integration**)
  - RS Forest based AAD (with streaming support **still needs better integration**)

To run the Isolation Forest / HS-Trees / RS-Forest based algorithms, the command has the following format:

    bash ./tree_aad.sh <dataset> <budget> <reruns> <tau> <inference_type> <query_type> <streaming[0|1]>

    for Isolation Forest set <inference_type>=7; 
    for HSTrees, set <inference_type>=11;
    for RSForest, set <inference_type>=12;

example (with Isolation Forest, non-streaming):

    bash ./tree_aad.sh toy2 35 1 0.03 7 1 0


Reference(s):
-------------
  - Das, S., Wong, W-K., Dietterich, T., Fern, A. and Emmott, A. (2016). Incorporating Expert Feedback into Active Anomaly Discovery in the Proceedings of the IEEE International Conference on Data Mining. (http://web.engr.oregonstate.edu/~wongwe/papers/pdf/ICDM2016.AAD.pdf)
  (https://github.com/shubhomoydas/aad/blob/master/overview/ICDM2016-AAD.pptx)

  - Das, S., Wong, W-K., Fern, A., Dietterich, T. and Siddiqui, A. (2017). Incorporating Feedback into Tree-based Anomaly Detection, KDD Interactive Data Exploration and Analytics (IDEA) Workshop.
  (http://poloclub.gatech.edu/idea2017/papers/p25-das.pdf)
  (https://github.com/shubhomoydas/pyaad/blob/master/presentations/IDEA17_slides.pptx)


Running the original (non-tree/LODA based) AAD Code:
----------------------------------------------------
For the most straightforward execution of the code, assume that we have the original datafile and another file that has anomaly scores from an ensemble of detectors. One example of these files (and their formats) can be found under the folder 'sampledata'.

The output will have two files: file '*-baseline.csv' shows the number of true anomalies detected with each iteration if we do not incorporate feedback; and the file '*-num_seen.csv' shows the number of true anomalies detected when we incorporate feedback.

Sample run command (AAD original (non-tree/LODA based)):
--------------------------------------------------------
python ./pyalad/alad.py --startcol=2 --labelindex=1 --header --randseed=42 --dataset=toy --datafile=./sampledata/toy.csv --scoresfile=./sampledata/toy_scores.csv --querytype=1 --inferencetype=3 --constrainttype=1 --sigma2=0.5 --reps=1 --reruns=1 --budget=10 --tau=0.03 --Ca=100 --Cn=1 --Cx=1000 --withprior --unifprior --runtype=simple --log_file=./temp/pyaad.log --resultsdir=./temp --ensembletype=regular --debug

R Version
---------
An older implementation in R is available at: https://github.com/shubhomoydas/aad
