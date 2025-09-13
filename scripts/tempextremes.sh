#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/auroraencoderanalysis"

# Path to catch errors
OUTPUT_FILE="${MODULE_DIR}/outputs/tempextremes.out"

# Data paths
STATIC_PATH="${MODULE_DIR}/data/static.nc"
ERA5_ZARR_PATH="https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

# Encoder variables
PATCH_SIZE="4"


#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

RUN_CMD="python scripts/tempextremes.py \
    --static-path ${STATIC_PATH} \
    --output-path ${OUTPUT_FILE} \
    --era5-zarr-path ${ERA5_ZARR_PATH} \
    --patch-size ${PATCH_SIZE}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
