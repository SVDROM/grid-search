from datetime import timedelta
import xarray as xr
import numpy as np
import pandas as pd


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
    the specified year and months. Note 29 Feb is skipped on leap
    years.

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

    dt = (np.unique(np.diff(data.time.values))).astype("timedelta64[h]").astype(int)
    if len(dt) > 1:
        msg = "The time axis of the input data must have uniform spacing."
        raise ValueError(msg)
    dt = dt[0]
    times = pd.date_range(
        f"{year}-01-01", f"{year}-12-31 23:00", freq=timedelta(hours=int(dt))
    )
    times = times[times.month.isin(months)]
    # drop 29 Feb for consistency
    times = times[~((times.month == 2) & (times.day == 29))]

    # handle 29 Feb on leap years
    doy = data.time.dt.dayofyear
    shift = data.time.dt.is_leap_year & (data.time.dt.month > 2)
    doy_corrected = xr.where(shift, doy - 1, doy)
    data = data.assign_coords(doy=("time", doy_corrected.data))
    data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))

    # keep only the requested months
    data = data.sel(time=data.time.dt.month.isin(months))
    data = data.compute()

    # now group by day of year and hour of day, and average
    # to compute climatology
    clim = data.groupby(["doy", "time.hour"]).mean()
    clim = clim.stack(time=("doy", "hour"))
    clim = clim.drop_vars(["doy", "hour"])
    return clim.assign_coords(time=times)
