#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/auroraencoderanalysis"
OUTPUT_DIR="${MODULE_DIR}/outputs"

# Path to catch errors
OUTPUT_FILE="${OUTPUT_DIR}/landsea.out"

# Data paths
EMBEDDINGS_PATH="gs://aurora-encoder-storage/encoder_embedding_20240713_20241821.zarr"
STATIC_PATH="gs://aurora-encoder-storage/static.zarr"

# Dates (end date needs to be at least 6 hours after start date)
START_DATE="2024-07-13T18:00:00"
END_DATE="2024-07-16T00:00:00"

# Test variables
TEST_LON_MIN=120
TEST_LON_MAX=210

# Encoder variables
PATCH_SIZE="4"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

RUN_CMD="python scripts/landsea.py \
    --embeddings-path ${EMBEDDINGS_PATH} \
    --static-path ${STATIC_PATH} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --patch-size ${PATCH_SIZE} \
    --test-lon-min ${TEST_LON_MIN} \
    --test-lon-max ${TEST_LON_MAX} \
    --output-dir ${OUTPUT_DIR}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
