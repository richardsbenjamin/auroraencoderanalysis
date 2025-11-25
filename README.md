# Exploring the Latent Space of Aurora's Encoder

## Getting Started
You'll need to set up the repository and environemnt on your machine. You'll need to make sure your machine has Git.

1.  Clone the repository:
    ```sh
    git clone https://github.com/richardsbenjamin/auroraencoderanalysis
    ```

2. Create the virtual environment. Conda is recommended for environemnt management (it's what we used). Once conda is installed, create the environment: 
    ```sh
    conda env create -f /path/to/auroraencoderanalysis/environments/auroralatent-1.0.yml
    ```
    activate the environment: 
    ```sh
    conda activate auroralatent-1.0
    ```

Most scripts only require a CPU as they do not rely on using the Aurora model. The `generate_embeddings.py` script generates the embeddings and was run using a A100 40GB GPU.  

## Generate embeddings. 

1. First, we must generate the dataset of latent vectors by running the script below. This requires the 40GB GPU:
    ```sh 
    ./scripts/generate_embeddings.sh
    ```

As this can take some time, you may be interested in skipping this step. The dataset has been made publicly available via `gs://aurora-encoder-storage/encoder_embedding_20240713_20241821.zarr`.

## Land-Sea Analysis
   
   1. To run the land-sea analysis:
      ```sh
      ./scripts/landsea.sh
      ```

## Extreme Temperature Analysis

1. Before running, you must ensure that you're able to download data via the CDA API, as the percentiles are downloaded during the script. To do this, you'll need to create an account, and store a .cdsapirc file in the root folder of your machine. Instructions [here](https://cds.climate.copernicus.eu/how-to-api).
   Once this is done, you can run:
    ```sh
    ./scripts/tempextremes.sh
    ```

## Atmospheric Instability

1. Before running, the atmospheric instability masks must be generated. 
    ```sh 
    ./scripts/calc_atmos_instability_mask.sh
    ```

    Again, this dataset has also been made publicly available `gs://aurora-encoder-storage/atmos_instability_masks.zarr`.
2. Then:
    ```sh 
    ./scripts/atmospheric_instability.sh
    ```
    
## Citation

To cite this work, please use the following BibTeX entry:

```bibtex
@misc{richards2025physicalconsistencyaurorasencoder,
    title={Physical Consistency of Aurora's Encoder: A Quantitative Study}, 
    author={Benjamin Richards and Pushpa Kumar Balan},
    year={2025},
    eprint={2511.07787},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2511.07787}, 
}
```
