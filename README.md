# SVDROM Grid Search

This project performs grid search experiments for training and evaluating Optimized Dynamic Mode Decomposition (OptDMD) models on climate data (the publicly available [ERA5 dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)).
The [SVD-ROM](https://github.com/SVDROM/svdrom) package is employed for performing OptDMD.
The main workflow is controlled by `train.py`, with configuration specified in `params.yaml`.

## Main Files

### `train.py`

- Trains OptDMD models for all combinations of parameters defined in `params.yaml`.
- Computes reconstruction and forecast RMSE for each model.
- Saves metrics and (optionally) models to disk.
- Uses Dask for parallel computation.
- Requires input SVD and scaler (containing the time average) files tracked by DVC.

### `params.yaml`

- Configuration file for grid search parameters.
- Key sections:
  - `dask`: Number of threads for parallel processing.
  - `train`: List of mode counts, Hankel preprocessing options, trial settings for parallel bagging.
  - `reconstruct`/`forecast`: Time ranges for evaluation.
  - `ins`: Paths to input data and models.
  - `outs`: Output directories for models and metrics.
  - `misc`: Additional settings (e.g., pressure level of the dataset).

## Usage

1. Ensure all input files (SVD, scaler, groundtruth) are available and tracked by DVC.
2. Adjust parameters in `params.yaml` as needed.
3. Run the training script from the command line:

   ```bash
   python train.py
   ```

4. Results (RMSE metrics and optionally models) are saved in the specified output directories.

## Results

Results are presented in the notebooks `evaluation_winter_2020.ipynb` and `evaluation_summer_2020.ipynb`, where an evaluation of 45 day DMD forecasts of winter and summer 2020 are presented.

## Requirements

- Python 3.10+
- Dask
- xarray
- numpy
- ruamel.yaml
- python-box
- svdrom
- DVC (for data tracking)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Folder Structure

- `input_data/`: Input datasets(tracked by DVC).
- `metrics/`: Output RMSE metrics.
- `models/`: Saved DMD models (if enabled).

## Notes

- Pressure level consistency is checked between `params.yaml` and DVC metadata.
- If the Dask dashboard link printed at runtime doesn't work, try replacing the hostname with `localhost` in the URL.

---
For more details, see `train.py` and `params.yaml`.
