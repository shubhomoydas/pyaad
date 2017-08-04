#!/bin/bash

# To run:
# bash ./if_aad.sh <dataset> <budget> <reruns> <tau>
#
# inferencetype(s):
#   1-Simple, 
#   2-Simple_Optim(same as simple), 
#   3-AATP with pairwise constraints,
#   6-ATGP (Iterative gradient)
#
# Examples:
# bash ./if_aad.sh toy 10 2 0.03


DATASET=$1
BUDGET=$2
RERUNS=$3
TAU=$4

UNIF_PRIOR_IND=1
QUERY_TYPE=1  # QUERY_DETERMINISIC
INFERENCE_TYPE=7  # 7 - AAD_IFOREST / 9 - IFOREST_ORIG
REPS=1  # number of independent data samples (input files)

# ==============================
# IForest types:
#   7 - AAD_IFOREST
#   9 - IFOREST_ORIG
# ------------------------------
INFERENCE_NAME="if_aad"
if [[ "$INFERENCE_TYPE" == "9" ]]; then
    INFERENCE_NAME="iforest"
elif [[ "$INFERENCE_TYPE" == "10" ]]; then
    INFERENCE_NAME="if_atgp"
fi

# ==============================
# Constraint types when Inference Type is AAD_PAIRWISE_CONSTR_UPD_TYPE
# ------------------------------
# AAD_CONSTRAINT_NONE = 0 (no constraints)
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

CA=100
CX=0.001
MAX_BUDGET=100
TOPK=0

MAX_ANOMALIES_CONSTRAINT=1000
MAX_NOMINALS_CONSTRAINT=1000

N_TREES=100
N_SAMPLES=256

# ==============================
# Iforest anomaly score computation for a node
# 0 - IFOR_SCORE_TYPE_INV_PATH_LEN
# 1 - IFOR_SCORE_TYPE_INV_PATH_LEN_EXP
# 3 - IFOR_SCORE_TYPE_CONST
# 4 - IFOR_SCORE_TYPE_NEG_PATH_LEN
# ------------------------------
IFOR_SCORE_TYPE=3

IFOR_LEAF_ONLY=0
if [[ "$IFOR_LEAF_ONLY" == "1" ]]; then
    IFOR_LEAF_ONLY="--ifor_add_leaf_nodes_only"
    IFOR_LEAF_ONLY_SIG="_leaf"
    IFOR_SCORE_TYPE=4  # only IFOR_SCORE_TYPE_NEG_PATH_LEN supported for leaf-only
else
    IFOR_LEAF_ONLY=""
    IFOR_LEAF_ONLY_SIG=""
fi

RAND_SEED=42

# ==============================
# Input CSV file properties.
# the startcol and labelindex are 1-indexed
# ------------------------------
STARTCOL=2
LABELINDEX=1

# SIGMA2 determines the weight on prior.
SIGMA2=0.5

# ==============================
# Following option determines whether we want to put a prior on weights.
# The type of prior is determined by option UNIF_PRIOR (defined later)
#   WITH_PRIOR="" - Puts no prior on weights
#   WITH_PRIOR="--withprior" - Adds prior to weights as determined by UNIF_PRIOR
# ------------------------------
WITH_PRIOR_IND=1
if [[ "$WITH_PRIOR_IND" == "1" ]]; then
    WITH_PRIOR="--withprior"
    WITH_PRIOR_SIG="_s${SIGMA2}"
else
    WITH_PRIOR=""
    WITH_PRIOR_SIG="_noprior"
fi

if [[ "$UNIF_PRIOR_IND" == "1" ]]; then
    UNIF_PRIOR="--unifprior"
else
    UNIF_PRIOR=""
fi

#------------------------------------------------------------------
# --runtype=[simple|multi]:
#    Whether the there are multiple sub-samples for the input dataset
#------------------------------------------------------------------
#RUN_TYPE=simple
RUN_TYPE=multi

BASE_DIR=./datasets
LOG_PATH=./temp
PYTHON_CMD=pythonw
# source virtualenv/python/bin/activate

SCRIPT_PATH=pyalad/isolation_forest_aad.py

DATASET_DIR="${BASE_DIR}/anomaly/$DATASET"


NAME_PREFIX="${INFERENCE_NAME}_trees${N_TREES}_samples${N_SAMPLES}_nscore${IFOR_SCORE_TYPE}${IFOR_LEAF_ONLY_SIG}_tau${TAU}${TAU_SIG}${WITH_PRIOR_SIG}_cx${CX}"
if [[ "$INFERENCE_TYPE" == "9" ]]; then
    NAME_PREFIX="${INFERENCE_NAME}_trees${N_TREES}_samples${N_SAMPLES}"
fi

LOG_FILE=$LOG_PATH/${NAME_PREFIX}_${DATASET}.log

ORIG_FEATURES_PATH=${DATASET_DIR}/fullsamples
DATA_FILE=${ORIG_FEATURES_PATH}/${DATASET}_1.csv

RESULTS_PATH="results/$DATASET/${NAME_PREFIX}"

mkdir -p "${LOG_PATH}"
mkdir -p "${RESULTS_PATH}"

MODEL_FILE=${LOG_PATH}/${NAME_PREFIX}.mdl
LOAD_MODEL="--load_model"
SAVE_MODEL="--save_model"

${PYTHON_CMD} ${SCRIPT_PATH} --startcol=$STARTCOL --labelindex=$LABELINDEX --header \
    --filedir=$ORIG_FEATURES_PATH --datafile=$DATA_FILE \
    --resultsdir=$RESULTS_PATH \
    --randseed=$RAND_SEED --dataset=$DATASET --querytype=$QUERY_TYPE \
    --inferencetype=$INFERENCE_TYPE --constrainttype=$CONSTRAINT_TYPE \
    --sigma2=$SIGMA2 --runtype=$RUN_TYPE --reps=$REPS --reruns=$RERUNS \
    --budget=$BUDGET --maxbudget=$MAX_BUDGET --topK=$TOPK \
    --tau=$TAU --ifor_n_trees=$N_TREES --ifor_n_samples=$N_SAMPLES \
    --ifor_score_type=${IFOR_SCORE_TYPE} ${IFOR_LEAF_ONLY} \
    --Ca=$CA --Cn=1 --Cx=$CX $WITH_PRIOR $UNIF_PRIOR \
    --max_anomalies_in_constraint_set=$MAX_ANOMALIES_CONSTRAINT \
    --max_nominals_in_constraint_set=$MAX_NOMINALS_CONSTRAINT \
    --log_file=$LOG_FILE --cachedir=$MODEL_PATH \
    --modelfile=${MODEL_FILE} ${LOAD_MODEL} ${SAVE_MODEL} \
    --debug
