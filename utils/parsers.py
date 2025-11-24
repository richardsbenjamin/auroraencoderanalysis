from __future__ import annotations

from argparse import ArgumentParser

from auroraencoderanalysis._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auroraencoderanalysis._typing import Namespace


ATMOS_INSTABILITY_PARSER_ARGS = {
    "mask-path": {
        "type": str,
        "required": True,
        "help": "Zarr path to the atmospheric instability mask.",
    },
    "start-date": {
        "type": str,
        "required": True,
        "help": "Start date for the analysis given in format YYY-MM-DD."
    },
    "end-date": {
        "type": str,
        "required": True,
        "help": "End date for the analysis given in format YYY-MM-DD."
    },
    "patch-size": {
        "type": int,
        "required": True,
        "help": "The patch size of the encoder."
    },
    "embeddings-path": {
        "type": str,
        "required": True,
        "help": "Zarr path to where the Aurora embeddings are stored."
    },
    "output-dir": {
        "type": str,
        "required": True,
        "help": "Output path where results are saved."
    },
}

ATMOS_INSTABILITY_MASK_PARSER_ARGS = {
    "levels-path": {
        "type": str,
        "required": True,
        "help": "Zarr path to ERA5 levels inputs file for the Aurora model."
    },
    "output-zarr": {
        "type": str,
        "required": True,
        "help": "Output zarr to save the mask."
    },
    "start-date": {
        "type": str,
        "required": True,
        "help": "Start date for the ERA5 dataset given in format YYY-MM-DD."
    },
    "end-date": {
        "type": str,
        "required": True,
        "help": "End date for the ERA5 dataset given in format YYY-MM-DD."
    },
}

GEN_EMBEDDINGS_PARSER_ARGS = {
    "static-path": {
        "type": str,
        "required": True,
        "help": "Path to static inputs file for the Aurora model."
    },
    "singles-path": {
        "type": str,
        "required": True,
        "help": "Zarr path to the ERA5 single level variables. Will likely throw error if not EDH."
    },
    "levels-path": {
        "type": str,
        "required": True,
        "help": "Zarr path to the ERA5 pressure level variables. Will likely throw error if not EDH."
    },
    "output-zarr-path": {
        "type": str,
        "required": True,
        "help": "Output zarr directory to store the generated embeddings."
    },
    "start-date": {
        "type": str,
        "required": True,
        "help": "Start date for the ERA5 dataset given in format YYY-MM-DD."
    },
    "end-date": {
        "type": str,
        "required": True,
        "help": "End date for the ERA5 dataset given in format YYY-MM-DD."
    },
    "patch-size": {
        "type": int,
        "required": True,
        "help": "The patch size of the encoder."
    },
    "embed-dim": {
        "type": int,
        "required": True,
        "help": "The embedded dimension."
    },
    "n-embed-levels": {
        "type": int,
        "required": True,
        "help": "The number of embedded pressure levels."
    },
}

LAND_SEA_PARSER_ARGS = {
    "embeddings-path": {
        "type": str,
        "required": True,
        "help": "Zarr path to where the Aurora embeddings are stored."
    },
    "static-path": {
        "type": str,
        "required": True,
        "help": "Path to static inputs file for the Aurora model."
    },
    "start-date": {
        "type": str,
        "required": True,
        "help": "Start date for the ERA5 dataset given in format YYY-MM-DD."
    },
    "end-date": {
        "type": str,
        "required": True,
        "help": "End date for the ERA5 dataset given in format YYY-MM-DD."
    },
    "patch-size": {
        "type": int,
        "required": True,
        "help": "The patch size of the encoder."
    },
    "test-lon-min": {
        "type": int,
        "required": True,
        "help": "The longitude value determining the beginning of the test region."
    },
    "test-lon-max": {
        "type": int,
        "required": True,
        "help": "The longitude value determining the end of the test region."
    },
    "output-dir": {
        "type": str,
        "required": True,
        "help": "Output path where results are saved."
    },
}

TEMP_EXTREMES_PARSER_ARGS = {
    "embeddings-path": {
        "type": str,
        "required": True,
        "help": "Zarr path to where the Aurora embeddings are stored."
    },
    "singles-path": {
        "type": str,
        "required": True,
        "help": "Zarr path to the ERA5 single level variables. Will likely throw error if not EDH."
    },
    "start-date": {
        "type": str,
        "required": True,
        "help": "Start date for the extreme events given in format YYY-MM-DD."
    },
    "end-date": {
        "type": str,
        "required": True,
        "help": "End date for the extreme events given in format YYY-MM-DD."
    },
    "percentile-year": {
        "type": str,
        "required": True,
        "help": "Year used for selecting percentiles from the CDSAPI."
    },
    "percentiles": {
        "type": str,
        "required": True,
        "help": "Comma separated percentile levels."
    },
    "patch-size": {
        "type": int,
        "required": True,
        "help": "The patch size of the encoder."
    },
    "output-dir": {
        "type": str,
        "required": True,
        "help": "Output path where results are saved."
    },
}


def get_arg_parser(description: str, args_dict: dict) -> Namespace:
    arg_parser = ArgumentParser(description=description)
    for arg, arg_params in args_dict.items():
        arg_parser.add_argument(
            f'--{arg}',
            type=arg_params["type"],
            required=arg_params["required"],
            help=arg_params["help"],
        )
    run_args = arg_parser.parse_args()
    return run_args

def get_atmos_instability_parser() -> Namespace:
    return get_arg_parser(
        description="Calculate the atmospheric instability mask for the given dates.",
        args_dict=ATMOS_INSTABILITY_PARSER_ARGS,
    )

def get_calc_atmos_instability_mask_parser() -> Namespace:
    return get_arg_parser(
        description="Calculate the atmospheric instability mask for the given dates.",
        args_dict=ATMOS_INSTABILITY_MASK_PARSER_ARGS,
    )

def get_gen_embeddings_parser() -> Namespace:
    return get_arg_parser(
        description="Generate encoder emebeddings for specific dates.",
        args_dict=GEN_EMBEDDINGS_PARSER_ARGS,
    )

def get_land_sea_parser() -> Namespace:
    return get_arg_parser(
        description="Execute land sea analysis on the encoder embeddings.",
        args_dict=LAND_SEA_PARSER_ARGS,
    )

def get_temp_extremes_parser() -> Namespace:
    return get_arg_parser(
        description="Execute land sea analysis on the encoder embeddings.",
        args_dict=TEMP_EXTREMES_PARSER_ARGS,
    )

