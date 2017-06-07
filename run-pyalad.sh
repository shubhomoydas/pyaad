#!/bin/bash

# To run:
# bash ./run-pyalad.sh <dataset> <inferencetype[1|2|3]> <reps[1-20]> <unif_prior[0|1]> <tau> <query_type> <base algo[if|loda|lof]>
#
# inferencetype(s):
#   1-Simple, 
#   2-Simple_Optim(same as simple), 
#   3-AATP with pairwise constraints,
#   6-ATGP (Iterative gradient)
#
# Examples:
# bash ./run-pyalad.sh abalone 3 1 1 0.01 1 none


DATASET=$1
INFERENCE_TYPE=$2
REPS=$3
UNIF_PRIOR_IND=$4
TAU=$5
QUERY_TYPE=$6

BASE_ALGO=$7 # "if" # "features" # "if"  # "loda"  # "lof"

# ==============================
# Constraint types when Inference Type is AAD_PAIRWISE_CONSTR_UPD_TYPE
# ------------------------------
# AAD_CONSTRAINT_PAIRWISE = 1 (slack vars [0, Inf]; weights [-Inf, Inf])
# AAD_CONSTRAINT_PAIRWISE_WEIGHTS_POSITIVE_SUM_1 = 2 (slack vars [0, Inf]; weights [0, Inf])
# AAD_CONSTRAINT_WEIGHTS_POSITIVE_SUM_1 = 3 (no pairwise; weights [0, Inf], sum(weights)=1)
# AAD_CONSTRAINT_TAU_INSTANCE = 4 (tau-th quantile instance will be used in pairwise constraints)
CONSTRAINT_TYPE=4
if [[ "$CONSTRAINT_TYPE" == "4" ]]; then
    TAU_SIG="_xtau"
else
    TAU_SIG=""
fi

# ==============================
# When only a subset of labeled instances are used for pair-wise
# constraints, then the subset might be selected using one of the strategies:
#   ORDER_BY_VIOLATED="" - select instances uniformly at random (separately for anomaly and nominal)
#   ORDER_BY_VIOLATED="--orderbyviolated" - select most violated instances (separately for anomaly and nominal)
# ------------------------------
ORDER_BY_VIOLATED_IND=0
if [[ "$ORDER_BY_VIOLATED_IND" == "1" ]]; then
    ORDER_BY_VIOLATED="--orderbyviolated"
    OBV_SIG="_obv"
else
    ORDER_BY_VIOLATED=""
    OBV_SIG=""
fi

# ==============================
# We may ignore the AATP hinge loss and only use the prior and constraints.
#   IGNORE_AATP_LOSS="" - adds AATP loss to optimization
#   IGNORE_AATP_LOSS="--ignoreAATPloss" - ignores AATP loss from optimization
# ------------------------------
IGNORE_AATP_LOSS_IND=0
if [[ "$IGNORE_AATP_LOSS_IND" == "1" ]]; then
    IGNORE_AATP_LOSS="--ignoreAATPloss"
    AATP_SIG="_noAATP"
else
    IGNORE_AATP_LOSS=""
    AATP_SIG=""
fi

CA=100
MAX_BUDGET=100
TOPK=0

RERUNS=10
BUDGET=60 # 100

MAX_ANOMALIES_CONSTRAINT=1000
MAX_NOMINALS_CONSTRAINT=1000

RAND_SEED=42

# ==============================
# Input CSV file properties.
# the startcol and labelindex are 1-indexed
# ------------------------------
STARTCOL=2
LABELINDEX=1

# ==============================
# Following option determines whether we want to put a prior on weights.
# The type of prior is determined by option UNIF_PRIOR (defined later)
#   WITH_PRIOR="" - Puts no prior on weights
#   WITH_PRIOR="--withprior" - Adds prior to weights as determined by UNIF_PRIOR
# ------------------------------
WITH_PRIOR_IND=1

if [[ "$INFERENCE_TYPE" == "3" ]]; then
    INFERENCE_NAME=aad
elif [ "$INFERENCE_TYPE" == "6" ]; then
    INFERENCE_NAME=atgp
    WITH_PRIOR_IND=0
    CA=1
elif [ "$INFERENCE_TYPE" == "8" ]; then
    INFERENCE_NAME="simple_pairwise"
    WITH_PRIOR_IND=1
    UNIF_PRIOR_IND=1
else
    INFERENCE_NAME=other
fi

if [[ "$WITH_PRIOR_IND" == "1" ]]; then
    WITH_PRIOR="--withprior"
    WITH_PRIOR_SIG=""
else
    WITH_PRIOR=""
    WITH_PRIOR_SIG="_noprior"
fi

# SIGMA2 determines the weight on prior.
SIGMA2=0.5

if [[ "$UNIF_PRIOR_IND" == "1" ]]; then
    UNIF_PRIOR="--unifprior"
else
    UNIF_PRIOR=""
fi

# ==============================
# Should we use a random instance as the tau-th instance in the first feedback cycle?
#   RAND_TAU_INST_START_SIG="" - uses true tau-th ranked instance
#   RAND_TAU_INST_START_SIG="--random_instance_at_start" uses a random instance as tau-th ranked instance in first feedback round
# ------------------------------
RAND_TAU_INST_START_IND=0
if [[ "$RAND_TAU_INST_START_IND" == "1" ]]; then
    RAND_TAU_INST_START="--random_instance_at_start"
    RAND_TAU_INST_START_SIG="_rndstart"
else
    RAND_TAU_INST_START=""
    RAND_TAU_INST_START_SIG=""
fi

# ==============================
# PSEUDO_OPT is *NOT* to be used at present.
# ------------------------------
PSEUDO_OPT=""  # "--pseudoanomrank_always"

#------------------------------------------------------------------
# --runtype=[simple|multi]:
#    Whether the there are multiple sub-samples for the input dataset
#------------------------------------------------------------------
#RUN_TYPE=simple
RUN_TYPE=multi

#------------------------------------------------------------------
# --ensembletype=[regular|loda]:
#    'regular' if the file has precomputed scores from ensembles
#    'loda' if LODA projections are to be used as ensemble members
#------------------------------------------------------------------
ENSEMBLE_TYPE=loda
#ENSEMBLE_TYPE=regular
if [[ "$RUN_TYPE" == "simple" ]]; then
    ENSEMBLE_TYPE=regular
fi

BASE_DIR=./datasets
LOG_PATH=./temp
# source virtualenv/python/bin/activate

OPR=1
if [[ "$OPR" == "4" ]]; then
    SCRIPT_PATH=pyalad/test_results_support.py
elif [[ "$OPR" == "5" ]]; then
    SCRIPT_PATH=pyalad/summarize_alad_results.py
else
    SCRIPT_PATH=pyalad/alad.py
fi

DATASET_DIR="${BASE_DIR}/anomaly/$DATASET"

LOG_FILE=$LOG_PATH/${INFERENCE_NAME}-${DATASET}_tau${TAU}_ha${MAX_ANOMALIES_CONSTRAINT}_hn${MAX_NOMINALS_CONSTRAINT}${TAU_SIG}${OBV_SIG}${AATP_SIG}${WITH_PRIOR_SIG}_${BASE_ALGO}${RAND_TAU_INST_START_SIG}.log

FEATURE_SET="featurebag_${BASE_ALGO}_50_anom_sm"
SCORES_DIR="${DATASET_DIR}/${FEATURE_SET}"
ORIG_FEATURES_PATH=${DATASET_DIR}/fullsamples
if [[ "$OPR" == "3" ]]; then
    RESULTS_PATH="${RESULTS_PATH}/ai2results"
else
    RESULTS_PATH="results/$DATASET/${INFERENCE_NAME}_sensitivity_tau${TAU}_ha${MAX_ANOMALIES_CONSTRAINT}_hn${MAX_NOMINALS_CONSTRAINT}${TAU_SIG}${OBV_SIG}${AATP_SIG}${WITH_PRIOR_SIG}${RAND_TAU_INST_START_SIG}"
    if [ "$INFERENCE_TYPE" == "6" ]; then
        # ATGP
        RESULTS_PATH="results/$DATASET/${INFERENCE_NAME}"
    fi
fi

MODEL_PATH="models/$DATASET"
if [[ "$MODEL_PATH" != "" && "$ENSEMBLE_TYPE" == "loda" ]]; then
    MODEL_PATH="${MODEL_PATH}/fullmodel/pyalad"
    mkdir -p "${MODEL_PATH}"
fi

PLOTS_PATH=${DATASET_DIR}/fullplots

mkdir -p "${LOG_PATH}"
mkdir -p "${RESULTS_PATH}"
mkdir -p "${PLOTS_PATH}"

#------------------------------------------------------------------
# --optimlib=[cvxopt|scipy]:
#    optimization library to use
#------------------------------------------------------------------
#OPTIM_LIB=cvxopt
OPTIM_LIB=scipy

#------------------------------------------------------------------
# --cachetype=[csv|pydata]:
#    Type of cache used to save model parameters
#------------------------------------------------------------------
CACHE_TYPE=pydata

MEAN_REL_LOSS=""  # "--withmeanrelativeloss"
BATCH=""  # "--batch"

#echo $DATASET
#echo $DATA_FILE
#echo $SCORES_FILE
#echo $ENSEMBLE_TYPE

python ${SCRIPT_PATH} --startcol=2 --labelindex=1 --header \
    --randseed=$RAND_SEED --dataset=$DATASET --querytype=$QUERY_TYPE \
    --inferencetype=$INFERENCE_TYPE --constrainttype=$CONSTRAINT_TYPE \
    ${ORDER_BY_VIOLATED} ${IGNORE_AATP_LOSS} \
    --sigma2=0.5 --runtype=$RUN_TYPE --reps=$REPS --reruns=$RERUNS \
    --budget=$BUDGET --maxbudget=$MAX_BUDGET --topK=$TOPK \
    --tau=$TAU --tau_nominal=0.03 ${RAND_TAU_INST_START} \
    --Ca=$CA --Cn=1 --Cx=1000 $WITH_PRIOR $UNIF_PRIOR \
    --max_anomalies_in_constraint_set=$MAX_ANOMALIES_CONSTRAINT \
    --max_nominals_in_constraint_set=$MAX_NOMINALS_CONSTRAINT \
    --relativeto=1 --query_search_candidates=3 --query_search_depth=3 \
    --cachetype=$CACHE_TYPE --optimlib=$OPTIM_LIB \
    --log_file=$LOG_FILE --filedir=$ORIG_FEATURES_PATH --cachedir=$MODEL_PATH \
    --resultsdir=$RESULTS_PATH --plotsdir=$PLOTS_PATH \
    --ensembletype=$ENSEMBLE_TYPE \
    --datafile=$DATA_FILE --scoresfile=$SCORES_FILE \
    --scoresdir=$SCORES_DIR \
    --debug
