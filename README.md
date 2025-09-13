# Exploring the Latent Space of Aurora's Encoder

## Getting Started
You'll need to set up the repository and environemnt on your machine. You'll need to make sure your machine has Git.

1.  Clone the repository:
    ```sh
    git clone https://github.com/richardsbenjamin/auroraencoderanalysis
    ```

2. Create the virtual environment. Conda is recommended for environemnt management (it's what we used). Code in this repo was developed using `conda 25.7.0`; install instructions found [here](https://docs.anaconda.com/miniconda/). Once conda is installed, create the environment: 
    ```sh
    conda env create -f /path/to/auroraencoderanalysis/environments/auroralatent-1.0.yml
    ```
    activate the environment: 
    ```sh
    conda activate auroralatent-1.0.yml
    ```

Since the scripts rely on running the Aurora encoder module, you will need around 40GB of RAM. Running the land sea analysis can be done on a CPU with sufficient memory because it only generates one embedding. However, the temperature extremes analysis will require a GPU as it generates several embeddings.

## Land Sea Analysis

You can run the land sea analysis by running the below: 

1. Make sure the environment you've just created is active: 

    ```sh
    conda activate dlesym-0.1
    ```

2.`./scripts/landsea.sh` 

## Temperature Extremes Analysis

Before running this analyis, you must ensure that you're able to download data via the CDA API, as the percentiles are downloaded during the script. To do this, you'll need to create an account, and store a .cdsapirc file in the root folder of your machine. Instructions [here](https://cds.climate.copernicus.eu/how-to-api).

Once this is done, you can run:
    ```sh
    ./scripts/tempextremes.sh
    ```
