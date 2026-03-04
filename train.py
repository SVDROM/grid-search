import pickle
import xarray as xr
import dask
import dask.array as da
import numpy as np
from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
from svdrom.dmd import OptDMD
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from utils import compute_rmse


yaml = YAML(typ="safe")

dask.config.set(scheduler="threads", num_workers=2)


def reconstruct(
    dmd: OptDMD,
    params: ConfigBox,
) -> xr.DataArray:
    """Given a fitted DMD model and the parameters object,
    produce a DMD reconstruction and compute the RMSE as a function
    of lead time.

    Parameters
    ----------
    dmd: svdrom.OptDMD
        The fitted OptDMD model.
    params: ConfigBox
        The parameters object, containing the requested reconstruction
        details.

    Returns
    -------
    xarray.DataArray
        The computed reconstruction RMSE averaged across latitude and
        longitude, as a function of time.
    """
    groundtruth = xr.open_dataarray(str(params.ins.groundtruth), chunks="auto")
    groundtruth = groundtruth.sel(
        time=slice(
            params.reconstruct.reconstruct_start, params.reconstruct.reconstruct_end
        )
    )

    with open(params.ins.scaler, "rb") as f:
        scaler = pickle.load(f)

    reconstruction = dmd.reconstruct(
        t=slice(
            params.reconstruct.reconstruct_start,
            params.reconstruct.reconstruct_end,
        )
    )
    if dmd.num_trials > 0:
        # if bagging has been applied, keep only the ensemble mean
        reconstruction = reconstruction[0]

    reconstruction = reconstruction.unstack()
    reconstruction = reconstruction.squeeze()
    mean = scaler.mean
    if isinstance(mean, xr.Dataset):
        mean = mean.to_dataarray().squeeze()
    reconstruction = reconstruction.copy(data=reconstruction + mean)

    return compute_rmse(groundtruth, reconstruction)


def forecast(
    dmd: OptDMD,
    params: ConfigBox,
) -> xr.DataArray:
    """Given a fitted DMD model and the parameters object,
    produce a DMD forecast and compute the RMSE as a function
    of time.

    Parameters
    ----------
    dmd: svdrom.OptDMD
        The fitted OptDMD model.
    params: ConfigBox
        The parameters object, containing the requested forecast
        details.

    Returns
    -------
    xarray.DataArray
        The computed forecast RMSE averaged across latitude and
        longitude, as a function of time.
    """
    forecast_end = datetime.strptime(
        params.forecast.forecast_start, "%Y-%m-%dT%H"
    ) + timedelta(days=params.forecast.forecast_days)
    forecast_end = forecast_end.strftime("%Y-%m-%dT%H")

    groundtruth = xr.open_dataarray(params.ins.groundtruth, chunks="auto")
    groundtruth = groundtruth.sel(
        time=slice(params.forecast.forecast_start, forecast_end)
    )

    with open(params.ins.scaler, "rb") as f:
        scaler = pickle.load(f)

    forecast = dmd.forecast(f"{forecast_end} D")

    if dmd.num_trials > 0:
        # if bagging has been applied, keep only the ensemble mean
        forecast = forecast[0]

    forecast = forecast.unstack()
    forecast = forecast.squeeze()
    mean = scaler.mean
    if isinstance(mean, xr.Dataset):
        mean = mean.to_dataarray().squeeze()
    forecast = forecast.copy(data=forecast + mean)

    return compute_rmse(groundtruth, forecast)


def main() -> None:
    """For all combinations of number of modes and whether to perform Hankel
    pre-processing or not (specified in params.yaml), train an OptDMD model and
    compute the reconstruction and forecast RMSE.
    """
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
    for n_modes in params.train.n_modes:
        for hankel in params.train.hankel:

            ###########################
            ### Train the DMD model ###
            ###########################
            print(
                "Computing DMD model for: "
                f"n_modes = {n_modes}, "
                f"hankel = {hankel}, "
                f"num_trials = {params.train.num_trials}, "
                f"trial_size = {params.train.trial_size}."
            )

            svd_path = params.ins.svd_hankel if hankel else params.ins.svd

            with open(svd_path, "rb") as f:
                svd = pickle.load(f)

            dmd = OptDMD(
                n_modes=n_modes,
                time_units=params.train.time_units,
                num_trials=params.train.num_trials,
                trial_size=params.train.trial_size,
                parallel_bagging=True,
            )

            dmd.fit(svd.u, svd.s, svd.v)

            print("Done.")

            print("Saving DMD model to disk.")

            model_name = f"dmd_{n_modes}"
            model_name += "_hankel.pkl" if hankel else ".pkl"

            if dmd.num_trials > 0:
                models_path = Path(params.outs.proba_models_dir)
            else:
                models_path = Path(params.outs.deter_models_dir)

            models_path.mkdir(parents=True, exist_ok=True)

            with open(models_path / model_name, "wb") as f:
                pickle.dump(dmd, f)

            print("Done.")

            ########################################
            ### Compute the reconstruction error ###
            ########################################

            print("Computing the reconstruction RMSE.")
            reconstruction_rmse = reconstruct(dmd, params)
            print("Done.")

            print("Saving the reconstruction RMSE to disk.")
            metric_name = f"reconstruction_rmse_{n_modes}"
            metric_name += "_hankel.nc" if hankel else ".nc"

            if dmd.num_trials > 0:
                metrics_path = Path(params.outs.proba_metrics_dir)
            else:
                metrics_path = Path(params.outs.deter_metrics_dir)

            metrics_path.mkdir(parents=True, exist_ok=True)

            reconstruction_rmse.to_netcdf(metrics_path / metric_name)
            print("Done.")

            ##################################
            ### Compute the forecast error ###
            ##################################

            print("Computing the forecast RMSE.")
            forecast_rmse = forecast(dmd, params)
            print("Done.")

            print("Saving the forecast RMSE to disk.")
            metric_name = f"forecast_rmse_{n_modes}"
            metric_name += "_hankel.nc" if hankel else ".nc"

            if dmd.num_trials > 0:
                metrics_path = Path(params.outs.proba_metrics_dir)
            else:
                metrics_path = Path(params.outs.deter_metrics_dir)

            metrics_path.mkdir(parents=True, exist_ok=True)

            forecast_rmse.to_netcdf(metrics_path / metric_name)
            print("Done.")


if __name__ == "__main__":
    main()
