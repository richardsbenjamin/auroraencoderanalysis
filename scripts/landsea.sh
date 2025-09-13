#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/auroraencoderanalysis"

# Path to catch errors
OUTPUT_FILE="${MODULE_DIR}/outputs/landsea.out"

# Data paths
STATIC_PATH="${MODULE_DIR}/data/static.nc"
ERA5_ZARR_PATH="https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

# Dates (end date needs to be at least 6 hours after start date)
START_DATE="2022-01-01T00:00"
END_DATE="2022-01-01T06:00"

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
    --static-path ${STATIC_PATH} \
    --output-path ${OUTPUT_FILE} \
    --era5-zarr-path ${ERA5_ZARR_PATH} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --patch-size ${PATCH_SIZE} \
    --test-lon-min ${TEST_LON_MIN} \
    --test-lon-max ${TEST_LON_MAX}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
