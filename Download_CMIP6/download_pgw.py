# %% [markdown]
# # This script is to download CMIP6 data using intake-esm library
#
# Read more from link
#
# https://intake-esm.readthedocs.io/en/stable/tutorials/loading-cmip6-data.html
#
# How to use intake to download GCM
#

# %%
import concurrent.futures
import re
from pathlib import Path
from typing import Iterable

import chunk_util
import intake
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.autonotebook import tqdm

# %%
# Information of all CMIP6 files that one can download from intake esm data store
url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"


# %%
def natural_sort(l: Iterable[str]) -> list[str]:
    """
    Sort names like r1i1p1f1, r1i2p1f1 in a natural (numeric) order.
    - r1: Realization (initial condition run),
    - i1: Initialization method,
    - p1: Physical parameters,
    - f1: External forcings.

    Numeric order means that r1i1p1f1 < r2i1p1f1 < r11i1p1f1.

    :param l: list of names to be sorted
    """

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


# %%
def download_files(
    catalog_url: str, sid: str, exp: str, var: str, start_year: int, end_year: int
):
    """
    Download files from the CMIP6 data store

    :param catalog_url: intake esm data store
    :param sid: source_id
    :param exp: experiment_id
    :param var: variable_id
    """

    catalog = intake.open_esm_datastore(catalog_url)
    models = catalog.search(
        experiment_id=exp,
        table_id="Amon",
        variable_id=var,
        source_id=sid,
    )
    # then one might get several files with the same conditions
    # r1: Realization (initial condition run)
    # i1: Initialization method
    # p1: Physical parameters
    # f1: External forcings

    # if no files exist then print out error
    if len(models.df) == 0:
        print("*** No data found for", var, exp, sid)
        return False

    member_ids = natural_sort(models.df.member_id.values)

    # get the first one only then seach again
    first_member_id = member_ids[0]
    first_member = catalog.search(
        experiment_id=exp,
        table_id="Amon",
        variable_id=var,
        source_id=sid,
        member_id=first_member_id,
    )

    # if no files exist then print out error
    if len(first_member.df) == 0:
        print(
            "*** This is impossible, there must be data for",
            var,
            exp,
            sid,
            member_ids[0],
        )
        return False

    odir = Path("Download") / sid / exp
    odir.mkdir(parents=True, exist_ok=True)

    def output_file_name(key):
        return odir / f"{var}_{key}_{first_member_id}_{start_year}_{end_year}.nc"

    try:
        # If all output files exist, skip
        if all((output_file_name(key)).exists() for key in first_member.keys()):
            return True

        datasets: dict[str, xr.Dataset] = first_member.to_dataset_dict(
            xarray_open_kwargs={"consolidated": True}, progressbar=False
        )

        for key, ds in datasets.items():
            # Must use isin because some dataset has time variable not monotonically increasing
            year_data = ds.sel(
                time=ds.time.dt.year.isin(np.arange(start_year, end_year + 1))
            )

            years = np.unique(year_data.time.dt.year.values)

            if len(years) == 0:
                print(
                    f"{sid}, {exp}, {var}, requested {start_year}-{end_year} data but no data found"
                )
                return False

            if years[0] != start_year or years[-1] != end_year:
                print(
                    f"{sid}, {exp}, {var}, requested {start_year}-{end_year} data "
                    f"but only {years[0]}-{years[-1]} data found"
                )
                return False

            month_mean = year_data.groupby("time.month").mean("time").squeeze(drop=True)

            ofile = output_file_name(key)
            tmp_ofile = ofile.with_suffix(".tmp.nc")

            # Compression
            encoding = {
                var_name: {
                    "zlib": True,
                    "complevel": 1,
                    "chunksizes": chunk_util.chunk_shape_nD(
                        data.shape, chunkSize=64 * 2**10
                    ),
                }
                for var_name, data in month_mean.data_vars.items()
            }

            # Save to temporary file first, and then rename to output file to
            # avoid regarding corrupted file due to sudden termination as
            # complete file.
            month_mean.to_netcdf(
                tmp_ofile, format="NETCDF4_CLASSIC", engine="netcdf4", encoding=encoding
            )
            tmp_ofile.rename(ofile)

    except Exception as e:
        print("*** Couldn't download", var, exp, sid, e)
        return False

    return True


# %%
# Example usage
# download_files(url, "EC-Earth3", "historical", "tas", 1995, 2014)
# download_files(url, "EC-Earth3", "historical", "ta", 1995, 2014)
# download_files(url, "MPI-ESM1-2-HR", "historical", "tas", 1995, 2014)


# %%
def download_data(
    source_ids: list[str],
    experiments: list[str],
    variables: list[str],
    start_year: int,
    end_year: int,
):
    status = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        status = []
        for sid in source_ids:
            for exp in experiments:
                for var in variables:
                    future = executor.submit(
                        download_files, url, sid, exp, var, start_year, end_year
                    )
                    futures.append(future)
                    status.append(
                        {
                            "source_id": sid,
                            "experiment": exp,
                            "variable": var,
                        }
                    )

        for future, stat in tqdm(zip(futures, status), total=len(futures)):
            try:
                success = future.result()
            except Exception as e:
                success = False
                print("*** Error:", e)

            stat["success"] = success

    return pd.DataFrame(status)


# %%
source_ids = [
    "EC-Earth3",
    "MIROC6",
    "MRI-ESM2-0",
    "ACCESS-CM2",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
]
experiments = ["ssp585"]
variables = ["tas", "ta", "ua", "va", "hur", "zg", "ts"]

historical_status = download_data(source_ids, ["historical"], variables, 1995, 2014)
ssp_status = download_data(source_ids, experiments, variables, 2045, 2064)

download_status = pd.concat([historical_status, ssp_status], ignore_index=True)
print(f"Successfully downloaded {download_status['success'].sum()} files")

failed_download = download_status.query("~success")
if not failed_download.empty:
    print("Couldn't download the following files")
    print(failed_download)
else:
    print("No failed download")
