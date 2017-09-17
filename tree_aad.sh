#!/bin/bash

# To run:
# bash ./tree_aad.sh <dataset> <budget> <reruns> <tau> <inference_type> <query_type> <streaming[0|1]>
#
# detector_type(s):
#   1-Simple, 
#   2-Simple_Optim(same as simple), 
#   3-AATP with pairwise constraints,
#   6-ATGP (Iterative gradient)
#
# Examples:
# bash ./tree_aad.sh toy2 35 1 0.03 7 1 0
# bash ./tree_aad.sh toy2 35 1 0.03 11 1 1
# bash ./tree_aad.sh toy2 35 1 0.03 12 1 1

DATASET=$1
BUDGET=$2
RERUNS=$3
TAU=$4

# ==============================
# Supported DETECTOR_TYPE:
# ------------------------------
#  7 - AAD_IFOREST
# 11 - AAD_HSTREES
# 12 - AAD_RSFOREST
# ------------------------------
DETECTOR_TYPE=$5

# ==============================
# Query types
# ------------------------------
# QUERY_DETERMINISIC = 1
# QUERY_BETA_ACTIVE = 2
# QUERY_QUANTILE = 3
# QUERY_RANDOM = 4
# QUERY_SEQUENTIAL = 5
# QUERY_GP = 6 (Gaussian Process)
# QUERY_SCORE_VAR = 7
# ------------------------------
QUERY_TYPE=$6

# ==============================
# Streaming Ind
# ------------------------------
# 0 - No streaming
# 1 - Streaming
# ------------------------------
STREAMING_IND=$7

UNIF_PRIOR_IND=1
REPS=1  # number of independent data samples (input files)

N_EXPLORE=20  # number of unlabeled top ranked instances to explore (if explore/explore)

# ==============================
# CONSTRAINT_TYPE:
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

CA=1  #100
CX=0.001
MAX_BUDGET=300
TOPK=0

MAX_ANOMALIES_CONSTRAINT=1000
MAX_NOMINALS_CONSTRAINT=1000

N_SAMPLES=256

N_JOBS=4  # Number of parallel threads

# ==============================
# FOREST_SCORE_TYPE:
# 0 - IFOR_SCORE_TYPE_INV_PATH_LEN
# 1 - IFOR_SCORE_TYPE_INV_PATH_LEN_EXP
# 3 - IFOR_SCORE_TYPE_CONST
# 4 - IFOR_SCORE_TYPE_NEG_PATH_LEN
# 5 - HST_SCORE_TYPE
# 6 - RSF_SCORE_TYPE
# 7 - RSF_LOG_SCORE_TYPE
# 8 - ORIG_TREE_SCORE_TYPE
# ------------------------------
# Supported DETECTOR_TYPE:
#    7 - AAD_IFOREST
#   11 - AAD_HSTREES
#   12 - AAD_RSFOREST
# ------------------------------
INFERENCE_NAME=
FOREST_SCORE_TYPE=3
N_TREES=100
MAX_DEPTH=7  #15  # 10
FOREST_LEAF_ONLY=1
if [[ "$DETECTOR_TYPE" == "7" ]]; then
    INFERENCE_NAME="if_aad"
    MAX_DEPTH=100
elif [[ "$DETECTOR_TYPE" == "11" ]]; then
    INFERENCE_NAME="hstrees"
    FOREST_SCORE_TYPE=5
    N_TREES=30
    CA=1
elif [[ "$DETECTOR_TYPE" == "12" ]]; then
    INFERENCE_NAME="rsforest"
    FOREST_SCORE_TYPE=6  #7
    N_TREES=30
    CA=1
fi

if [[ "$FOREST_LEAF_ONLY" == "1" ]]; then
    FOREST_LEAF_ONLY="--forest_add_leaf_nodes_only"
    FOREST_LEAF_ONLY_SIG="_leaf"
    if [[ "$DETECTOR_TYPE" == "7" ]]; then
        # IFOR_SCORE_TYPE_NEG_PATH_LEN supported only for isolation forest leaf-only
        FOREST_SCORE_TYPE=4
    fi
else
    FOREST_LEAF_ONLY=""
    FOREST_LEAF_ONLY_SIG=""
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

STREAM_WINDOW=512
ALLOW_STREAM_UPDATE=
ALLOW_STREAM_UPDATE_SIG=
ALLOW_STREAM_UPDATE_IND=1
if [[ "$ALLOW_STREAM_UPDATE_IND" == "1" ]]; then
    ALLOW_STREAM_UPDATE="--allow_stream_update"
    ALLOW_STREAM_UPDATE_SIG="asu"
fi

if [[ "$STREAMING_IND" == "1" ]]; then
    STREAMING="--streaming"
    STREAMING_SIG="_stream"
    STREAMING_FLAGS="${STREAM_WINDOW}${ALLOW_STREAM_UPDATE_SIG}"
    PYSCRIPT=forest_aad_stream.py
else
    STREAMING=""
    STREAMING_SIG=
    STREAMING_FLAGS=
    PYSCRIPT=forest_aad_batch.py
fi

#------------------------------------------------------------------
# --runtype=[simple|multi]:
#    Whether the there are multiple sub-samples for the input dataset
#------------------------------------------------------------------
#RUN_TYPE=simple
RUN_TYPE=multi

NAME_PREFIX="${INFERENCE_NAME}_trees${N_TREES}_samples${N_SAMPLES}_i${DETECTOR_TYPE}_q${QUERY_TYPE}_bd${BUDGET}_nscore${FOREST_SCORE_TYPE}${FOREST_LEAF_ONLY_SIG}_tau${TAU}${TAU_SIG}${WITH_PRIOR_SIG}_ca${CA}_cx${CX}_d${MAX_DEPTH}${STREAMING_SIG}${STREAMING_FLAGS}"
if [[ "$DETECTOR_TYPE" == "9" ]]; then
    NAME_PREFIX="${INFERENCE_NAME}_trees${N_TREES}_samples${N_SAMPLES}"
fi

SCRIPT_PATH=./pyalad/${PYSCRIPT}
# personal laptop
BASE_DIR=./datasets${STREAMING_SIG}
LOG_PATH=./temp
PYTHON_CMD=pythonw
RESULTS_PATH="temp/$DATASET/${NAME_PREFIX}"

DATASET_DIR="${BASE_DIR}/anomaly/$DATASET"

LOG_FILE=$LOG_PATH/${NAME_PREFIX}_${DATASET}.log

ORIG_FEATURES_PATH=${DATASET_DIR}/fullsamples
DATA_FILE=${ORIG_FEATURES_PATH}/${DATASET}_1.csv

mkdir -p "${LOG_PATH}"
mkdir -p "${RESULTS_PATH}"

MODEL_FILE=${LOG_PATH}/${NAME_PREFIX}.mdl
LOAD_MODEL=  # "--load_model"
SAVE_MODEL=  # "--save_model"

PLOT2D=  #"--plot2D"

${PYTHON_CMD} ${SCRIPT_PATH} --startcol=$STARTCOL --labelindex=$LABELINDEX --header \
    --filedir=$ORIG_FEATURES_PATH --datafile=$DATA_FILE \
    --resultsdir=$RESULTS_PATH \
    --randseed=$RAND_SEED --dataset=$DATASET --querytype=$QUERY_TYPE \
    --detector_type=$DETECTOR_TYPE --constrainttype=$CONSTRAINT_TYPE \
    --sigma2=$SIGMA2 --runtype=$RUN_TYPE --reps=$REPS --reruns=$RERUNS \
    --budget=$BUDGET --maxbudget=$MAX_BUDGET --topK=$TOPK \
    --tau=$TAU --forest_n_trees=$N_TREES --forest_n_samples=$N_SAMPLES \
    --forest_score_type=${FOREST_SCORE_TYPE} ${FOREST_LEAF_ONLY} \
    --forest_max_depth=${MAX_DEPTH} \
    --Ca=$CA --Cn=1 --Cx=$CX $WITH_PRIOR $UNIF_PRIOR \
    --max_anomalies_in_constraint_set=$MAX_ANOMALIES_CONSTRAINT \
    --max_nominals_in_constraint_set=$MAX_NOMINALS_CONSTRAINT \
    --n_explore=${N_EXPLORE} \
    --log_file=$LOG_FILE --cachedir=$MODEL_PATH \
    --modelfile=${MODEL_FILE} ${LOAD_MODEL} ${SAVE_MODEL} \
    ${STREAMING} ${ALLOW_STREAM_UPDATE} --stream_window=${STREAM_WINDOW} ${PLOT2D} \
    --debug
