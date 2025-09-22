#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/auroraencoderanalysis"

# Path to catch errors
OUTPUT_FILE="${MODULE_DIR}/outputs/calc_atmos_instability_mask.out"

# Data paths
LEVELS_PATH="https://data.earthdatahub.destine.eu/era5/reanalysis-era5-pressure-levels-v0.zarr"
OUTPUT_ZARR_PATH="gs://aurora-encoder-storage/atmos_instability_masks.zarr"

# Dates
START_DATE="2024-07-01T06:00"
END_DATE="2024-08-21T00:00"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

RUN_CMD="python scripts/calc_atmos_instability_mask.py \
    --levels-path ${LEVELS_PATH} \
    --output-zarr ${OUTPUT_ZARR_PATH} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE}"


# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
