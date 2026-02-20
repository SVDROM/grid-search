import pickle
import xarray as xr
import dask
import dask.array as da
import numpy as np
from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
from svdrom.dmd import OptDMD
from dvclive import Live
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from utils import compute_rmse


yaml = YAML(typ="safe")

dask.config.set(scheduler="threads", num_workers=2)


def evaluate(
    dmd: OptDMD,
    reconstruct_start: str,
    reconstruct_end: str,
    groundtruth_path: Path = Path("input_data/era5_slice.zarr"),
    scaler_path: Path = Path("input_data/scaler.pkl"),
) -> xr.DataArray:

    groundtruth = xr.open_dataarray(str(groundtruth_path), chunks="auto")
    groundtruth = groundtruth.sel(time=slice(reconstruct_start, reconstruct_end))

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    reconstruction = dmd.reconstruct(t=slice(reconstruct_start, reconstruct_end))
    reconstruction = reconstruction.unstack()
    reconstruction = reconstruction.squeeze()
    mean = scaler.mean
    if isinstance(mean, xr.Dataset):
        mean = mean.to_dataarray().squeeze()
    reconstruction = reconstruction.copy(data=reconstruction + mean)

    return compute_rmse(groundtruth, reconstruction)


def forecast(
    dmd: OptDMD,
    forecast_start: str,
    forecast_days: int = 45,
    groundtruth_path: Path = Path("input_data/era5_slice.zarr"),
    scaler_path: Path = Path("input_data/scaler.pkl"),
) -> xr.DataArray:
    
    forecast_end = datetime.strptime(forecast_start, "%Y-%m-%dT%H") + timedelta(days=forecast_days)
    forecast_end = forecast_end.strftime("%Y-%m-%dT%H")

    groundtruth = xr.open_dataarray(str(groundtruth_path), chunks="auto")
    groundtruth = groundtruth.sel(time=slice(forecast_start, forecast_end))

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    forecast = dmd.forecast(f"{forecast_end} D")
    forecast = forecast.unstack()
    forecast = forecast.squeeze()
    mean = scaler.mean
    if isinstance(mean, xr.Dataset):
        mean = mean.to_dataarray().squeeze()
    forecast = forecast.copy(data=forecast + mean)

    return compute_rmse(groundtruth, forecast)


    svd_path = (
        Path("input_data/svd_hankel.pkl")
        if params.model.hankel
        else Path("input_data/svd.pkl")
    )

    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)
    models_dir = (
        models_dir / "deterministic"
        if params.model.num_trials == 0
        else models_dir / "probabilistic"
    )
    models_dir.mkdir(exist_ok=True)

    with Live("results/train") as live:
        # load the computed SVD
        with open(svd_path, "rb") as f:
            svd = pickle.load(f)

        # fit the DMD model
        dmd = OptDMD(
            n_modes=params.model.n_modes,
            time_units="h",
            num_trials=params.model.num_trials,
            trial_size=params.model.trial_size,
            parallel_bagging=True,
        )
        dmd.fit(svd.u, svd.s, svd.v)

        # evaluate the fitted DMD model
        rmse = evaluate(
            dmd,
            reconstruct_start=params.model.reconstruct_start,
            reconstruct_end=params.model.reconstruct_end,
        )
        rmse_mean = rmse.mean().values

        # log the model
        model_name = (
            f"dmd_{params.model.n_modes}_hankel.pkl"
            if params.model.hankel
            else f"dmd_{params.model.n_modes}.pkl"
        )
        desc = (
            "A DMD model fitted to atmospheric temperature at "
            f"{params.model.pressure_level} hPa."
        )
        live.log_artifact(
            str(models_dir / model_name),
            type="model",
            desc=desc,
            meta={
                "n_modes": params.model.n_modes,
                "hankel": params.model.hankel,
                "num_trials": params.model.num_trials,
            },
        )

        # log a plot of RMSE vs time
        fig, ax = plt.subplots()
        rmse.plot(ax=ax)
        ax.set_title(f"level = {params.model.pressure_level} hPa")
        ax.set_ylabel("Temperature RMSE [K]")
        ax.set_xlabel("Time")
        fig_name = (
            f"recon_{params.model.n_modes}_hankel.png"
            if params.model.hankel
            else f"recon_{params.model.n_modes}.png"
        )
        live.log_image(fig_name, fig)

        # log the mean RMSE as a metric
        live.log_metric("rmse", rmse_mean)


if __name__ == "__main__":
    main()
