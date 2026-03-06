from datetime import timedelta
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


def compute_climatology(
    data: xr.DataArray,
    year: int,
    months: list[int],
) -> xr.DataArray:
    """Given observed data, compute a climatological forecast over
    the specified year and months.

    Examples
    --------
    Given observed data for 2016-2019, compute the climatological
    forecast for Jan and Feb 2020:

    >>> climatology = compute_climatology(
            ground_truth.sel(time=slice("2016-01-01", "2019-12-31")),
            year=2020,
            months=[1, 2],
        )
    """
    months = sorted(months)
    climatology = data.sel(time=data.time.dt.month.isin(months))
    climatology = climatology.compute()
    climatology = climatology.sel(
        time=~((climatology.time.dt.month == 2) & (climatology.time.dt.day == 29))
    )  # drop 29th Feb for consistency
    climatology = climatology.groupby(["time.dayofyear", "time.hour"]).mean()
    hours = climatology.hour.values
    days = climatology.dayofyear.values
    t = []
    for d in days:
        for h in hours:
            dt = timedelta(days=int(d - 1), hours=int(h))
            t.append(np.timedelta64(dt, "h"))
    t = np.array(t)
    month = months[0]
    month = f"0{month}" if month < 10 else str(month)
    t = np.datetime64(f"{year}-{month}-01T00:00") + t
    climatology = climatology.stack(time=("dayofyear", "hour"))
    climatology = climatology.drop_vars(["time", "dayofyear", "hour"])
    return climatology.assign_coords(time=t)
