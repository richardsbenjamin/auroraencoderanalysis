#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/auroraencoderanalysis"
OUTPUT_DIR="${MODULE_DIR}/outputs"

# Path to catch errors
OUTPUT_FILE="${OUTPUT_DIR}/atmos_instability.out"

# Data paths
MASK_PATH="gs://aurora-encoder-storage/atmos_instability_masks.zarr"
EMBEDDINGS_PATH="gs://aurora-encoder-storage/encoder_embedding_20240713_20241821.zarr"

# Dates
START_DATE="2024-07-13T18:00:00"
END_DATE="2024-07-16T00:00:00"

# Encoder variables
PATCH_SIZE="4"


#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

RUN_CMD="python scripts/atmospheric_instability.py \
    --mask-path ${MASK_PATH} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --patch-size ${PATCH_SIZE} \
    --embeddings-path ${EMBEDDINGS_PATH} \
    --output-dir ${OUTPUT_DIR}"


# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
