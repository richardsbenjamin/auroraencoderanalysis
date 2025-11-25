#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/auroraencoderanalysis"
OUTPUT_DIR="${MODULE_DIR}/outputs"

# Path to catch errors
OUTPUT_FILE="${OUTPUT_DIR}/tempextremes.out"

# Data paths
EMBEDDINGS_PATH="gs://aurora-encoder-storage/encoder_embedding_20240713_20241821.zarr"
ERA5_SINGLES_PATH="https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"

# Dates (end date needs to be at least 6 hours after start date)
START_DATE="2024-07-13T18:00:00"
END_DATE="2024-07-18T18:00:00"

# Other variables
PERCENTILE_YEAR="2020-01-01"
PERCENTILES="75,90,95,99"
PATCH_SIZE="4"


cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

RUN_CMD="python scripts/tempextremes.py \
    --embeddings-path ${EMBEDDINGS_PATH} \
    --singles-path ${ERA5_SINGLES_PATH} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --percentile-year ${PERCENTILE_YEAR} \
    --percentiles ${PERCENTILES} \
    --patch-size ${PATCH_SIZE} \
    --output-dir ${OUTPUT_DIR}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
