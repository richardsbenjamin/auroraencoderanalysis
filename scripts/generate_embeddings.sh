#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/auroraencoder"

# Path to catch errors
OUTPUT_FILE="${MODULE_DIR}/outputs/generate_embeddings.out"

# Data paths
STATIC_PATH="${MODULE_DIR}/data/static.nc"
SINGLES_PATH="https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
LEVELS_PATH="https://data.earthdatahub.destine.eu/era5/reanalysis-era5-pressure-levels-v0.zarr"
OUTPUT_ZARR_PATH="gs://aurora-encoder-storage/test_run.zarr"

# Dates
START_DATE="2022-01-01T00:00"
END_DATE="2022-01-01T06:00"

# Encoder variables
PATCH_SIZE="4"
EMBED_DIM="512"
N_EMBED_LEVELS="3"


#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

RUN_CMD="python scripts/generate_embeddings.py \
    --static-path ${STATIC_PATH} \
    --singles-path ${SINGLES_PATH} \
    --levels-path ${LEVELS_PATH} \
    --output-zarr-path ${OUTPUT_ZARR_PATH} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --patch-size ${PATCH_SIZE} \
    --embed-dim ${EMBED_DIM} \
    --n-embed-levels ${N_EMBED_LEVELS}"


# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi