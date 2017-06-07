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


This codebase has two different algorithms:
  - The LODA based AAD, which was the earlier work
  - The Isolation Forest based AAD -- this is the most recent work. Expect documentation on this by end of July.

To run the Isolation Forest based code, see rough-if_aad.txt. A sample command is:
  bash ./run-if_aad.sh toy2 35 1 0.03

Reference:
----------
Das, S., Wong, W-K., Dietterich, T., Fern, A. and Emmott, A. (2016). Incorporating Expert Feedback into Active Anomaly Discovery. To appear in the Proceedings of the IEEE International Conference on Data Mining. (http://web.engr.oregonstate.edu/~wongwe/papers/pdf/ICDM2016.AAD.pdf)

Running the AAD Code:
---------------------
For the most straightforward execution of the code, assume that we have the original datafile and another file that has anomaly scores from an ensemble of detectors. One example of these files (and their formats) can be found under the folder 'sampledata'.

The output will have two files: file '*-baseline.csv' shows the number of true anomalies detected with each iteration if we do not incorporate feedback; and the file '*-num_seen.csv' shows the number of true anomalies detected when we incorporate feedback.

Sample run command (AAD):
-------------------------
python pyalad/alad.py --startcol=2 --labelindex=1 --header --randseed=42 --dataset=toy --datafile=/Users/moy/work/git/pyaad/sampledata/toy.csv --scoresfile=/Users/moy/work/git/pyaad/sampledata/toy_scores.csv --querytype=1 --inferencetype=3 --constrainttype=1 --sigma2=0.5 --reps=1 --reruns=1 --budget=10 --tau=0.03 --Ca=100 --Cn=1 --Cx=1000 --withprior --unifprior --runtype=simple --log_file=/Users/moy/work/temp/pyaad.log --resultsdir=/Users/moy/work/temp --ensembletype=regular --debug

R Version
--------------------
An older implementation in R is available at: https://github.com/shubhomoydas/aad
