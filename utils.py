import xarray as xr
import numpy as np


def compute_rmse(
    truth: xr.DataArray,
    prediction: xr.DataArray,
    prediction_var: xr.DataArray | None = None,
    lat_weighting: bool = True,
    dim: str | tuple[str, ...] = ("latitude", "longitude"),
) -> xr.DataArray:
    """Computes the Root Mean Squared Error (RMSE), averaging
    along the specified dimension(s).
    If the prediction variance is provided, then the expected RMSE
    is computed, where 'prediction' and 'prediction_var' are taken as
    the mean and variance of a probabilistic forecast.
    A weighting function is applied by default so that spatial locations close
    to the equator receive a larger weight than those close to the poles.
    """
    prediction = prediction.real  # keep only real part of the prediction
    if isinstance(prediction_var, xr.DataArray):
        rmse = truth.copy(data=(truth - prediction) ** 2 + prediction_var)
    else:
        rmse = truth.copy(data=(truth - prediction) ** 2)
    if lat_weighting:
        lat_weights = np.cos(np.deg2rad(truth.latitude.values))
        lat_weights = lat_weights / np.mean(lat_weights)
        lat_weights = np.reshape(lat_weights, (1, len(lat_weights), 1))
        rmse *= lat_weights
    rmse = rmse.mean(dim=dim)
    rmse = rmse.clip(min=0)
    rmse = xr.ufuncs.sqrt(rmse)
    return rmse.compute()
