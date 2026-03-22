import pickle
import xarray as xr
from dask.distributed import Client
import dask.array as da
import numpy as np
from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
from svdrom.dmd import OptDMD
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from svdrom.weather_utils import compute_rmse


def get_reconstruction_rmse(
    dmd: OptDMD,
) -> xr.DataArray:
    """Given a fitted DMD model, produce a DMD reconstruction
    and compute the RMSE as a function of time.

    Parameters
    ----------
    dmd: svdrom.OptDMD
        The fitted OptDMD model.

    Returns
    -------
    xarray.DataArray
        The computed reconstruction RMSE averaged across latitude and
        longitude, as a function of time.
    """
    groundtruth_path = Path(params.ins.groundtruth)
    check_pressure_level(groundtruth_path)
    groundtruth = xr.open_dataarray(groundtruth_path, chunks="auto")
    groundtruth = groundtruth.sel(
        time=slice(
            params.reconstruct.reconstruct_start, params.reconstruct.reconstruct_end
        )
    )

    scaler_path = Path(params.ins.scaler)
    check_pressure_level(scaler_path)
    with open(scaler_path, "rb") as f:
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


def get_forecast_rmse(
    dmd: OptDMD,
) -> xr.DataArray:
    """Given a fitted DMD model, produce a DMD forecast
    and compute the RMSE as a function of time.

    Parameters
    ----------
    dmd: svdrom.OptDMD
        The fitted OptDMD model.

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

    groundtruth_path = Path(params.ins.groundtruth)
    check_pressure_level(groundtruth_path)
    groundtruth = xr.open_dataarray(groundtruth_path, chunks="auto")
    groundtruth = groundtruth.sel(
        time=slice(params.forecast.forecast_start, forecast_end)
    )

    scaler_path = Path(params.ins.scaler)
    check_pressure_level(scaler_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    forecast = dmd.forecast(f"{params.forecast.forecast_days} D")

    if dmd.num_trials > 0:
        # if bagging has been applied, keep only the ensemble mean
        forecast = forecast[0]

    forecast = forecast.unstack()
    forecast = forecast.squeeze()

    # make sure groundtruth and forecast have exactly the same time vector
    _, ind1, ind2 = np.intersect1d(
        groundtruth.time.values, forecast.time.values, return_indices=True
    )
    groundtruth = groundtruth.isel(time=ind1)
    forecast = forecast.isel(time=ind2)

    mean = scaler.mean
    if isinstance(mean, xr.Dataset):
        mean = mean.to_dataarray().squeeze()
    forecast = forecast.copy(data=forecast + mean)

    return compute_rmse(groundtruth, forecast)


def check_pressure_level(dataset_path: Path) -> None:
    """Check that the pressure level requested in params.yaml matches
    the pressure level of the target dataset. The target dataset must
    be tracked with DVC and the pressure level must be specified in the
    'meta' section of the corresponding .dvc file. The function raises
    ValueError if the pressure level does not match.

    Parameters
    ----------
    dataset_path: pathlib.Path
        Path of the target dataset, which must be tracked with DVC.
    """
    dvc_path = dataset_path.with_suffix(dataset_path.suffix + ".dvc")
    dvc_config = ConfigBox(yaml.load(open(dvc_path, encoding="utf-8")))
    pressure_level = dvc_config.outs[0].meta.pressure_level
    if pressure_level != params.misc.pressure_level:
        msg = (
            f"The pressure level specified in params.yaml: {params.misc.pressure_level} "
            f"does not match the pressure level in {str(dvc_path)}. "
            "Make sure you have checked out the correct dataset."
        )
        raise ValueError(msg)


def main() -> None:
    """For all combinations of number of modes and whether to perform Hankel
    pre-processing or not (specified in params.yaml), train an OptDMD model and
    compute the reconstruction and forecast RMSE.
    """
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
            svd_path = Path(svd_path)

            check_pressure_level(svd_path)

            with open(svd_path, "rb") as f:
                svd = pickle.load(f)

            dmd = OptDMD(
                n_modes=n_modes,
                time_units=params.train.time_units,
                num_trials=params.train.num_trials,
                trial_size=params.train.trial_size,
                parallel_bagging=True,
                seed=params.train.seed,
            )

            dmd.fit(svd.u, svd.s, svd.v)

            print("Done.")

            if params.outs.save_models:
                print("Saving DMD model to disk.")

                model_name = f"dmd_0{n_modes}" if n_modes < 10 else f"dmd_{n_modes}"
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
            reconstruction_rmse = get_reconstruction_rmse(dmd)
            print("Done.")

            print("Saving the reconstruction RMSE to disk.")
            metric_name = "reconstruction_rmse_"
            metric_name += f"0{n_modes}" if n_modes < 10 else f"{n_modes}"
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
            forecast_rmse = get_forecast_rmse(dmd)
            print("Done.")

            print("Saving the forecast RMSE to disk.")
            metric_name = f"forecast_rmse_"
            metric_name += f"0{n_modes}" if n_modes < 10 else f"{n_modes}"
            metric_name += "_hankel.nc" if hankel else ".nc"

            if dmd.num_trials > 0:
                metrics_path = Path(params.outs.proba_metrics_dir)
            else:
                metrics_path = Path(params.outs.deter_metrics_dir)

            metrics_path.mkdir(parents=True, exist_ok=True)

            forecast_rmse.to_netcdf(metrics_path / metric_name)
            print("Done.")


if __name__ == "__main__":
    yaml = YAML(typ="safe")
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    # set up a local multi-threaded Dask cluster
    client = Client(processes=False, threads_per_worker=params.dask.num_threads)
    print(f"Dask dashboard: {client.dashboard_link}")

    main()

    client.close()
