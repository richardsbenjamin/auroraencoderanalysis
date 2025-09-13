from argparse import ArgumentParser


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
    "static-path": {
        "type": str,
        "required": True,
        "help": "Path to static inputs file for the Aurora model."
    },
    "output-path": {
        "type": str,
        "required": True,
        "help": "Output directory to save images and output data."
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
    "era5-zarr-path": {
        "type": str,
        "required": True,
        "help": "The zarr path for the ERA5 data."
    },
    "patch-size": {
        "type": int,
        "required": True,
        "help": "The patch size of the encoder."
    },
    "test-lon-min": {
        "type": int,
        "required": True,
        "help": "The minimum longitude to define the test region for the logistic regression."
    },
    "test-lon-max": {
        "type": int,
        "required": True,
        "help": "The maximum longitude to define the test region for the logistic regression."
    },

}


def get_arg_parser(description: str, args_dict: dict) -> ArgumentParser:
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

def get_gen_embeddings_parser() -> ArgumentParser:
    return get_arg_parser(
        description="Generate encoder emebeddings for specific dates.",
        args_dict=GEN_EMBEDDINGS_PARSER_ARGS,
    )

def get_land_sea_parser() -> ArgumentParser:
    return get_arg_parser(
        description="Execute land sea analysis on the encoder embeddings.",
        args_dict=LAND_SEA_PARSER_ARGS,
    )

