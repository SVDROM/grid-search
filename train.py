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

from utils import compute_rmse


yaml = YAML(typ="safe")

dask.config.set(scheduler="threads", num_workers=2)


def evaluate(
    dmd: OptDMD,
    groundtruth_path: Path = Path("input_data/era5_slice.zarr"),
    scaler_path: Path = Path("input_data/scaler.pkl"),
    reconstruct_start: str = "2019-11-01T00",
    reconstruct_end: str = "2019-12-31T18",
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


def main() -> None:
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

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
            reconstruct_start="2019-11-01T00",
            reconstruct_end="2019-12-31T20",
        )
        rmse_mean = rmse.mean().values

        # log the model
        model_name = (
            f"dmd_{params.model.n_modes}_hankel.pkl"
            if params.model.hankel
            else f"dmd_{params.model.n_modes}.pkl"
        )
        live.log_artifact(
            str(models_dir / model_name),
            type="model",
            desc="A DMD model fitted to 10 years of atmospheric temperature at 850 hPa.",
            meta={
                "n_modes": params.model.n_modes,
                "hankel": params.model.hankel,
                "num_trials": params.model.num_trials,
            },
        )

        # log a plot of RMSE vs time
        fig, ax = plt.subplots()
        rmse.plot(ax=ax)
        ax.set_title("level = 850 hPa")
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
