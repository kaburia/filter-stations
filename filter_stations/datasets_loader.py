import xarray as xr
import pandas as pd
import numpy as np
import os
import glob
import rioxarray
import warnings
from huggingface_hub import HfFileSystem, hf_hub_url

class RainLoader:
    def __init__(self, source_path='', use_hf=True, repo_id='kaburia/DSAIL-weather-data', token=None):
        self.use_hf = use_hf
        self.repo_id = repo_id
        self.token = token
        self.source_path = source_path

        if use_hf:
            if not repo_id:
                raise ValueError("repo_id is required when use_hf=True")
            self.fs = HfFileSystem(token=token)

            if token:
                os.environ['GDAL_HTTP_HEADERS'] = f"Authorization: Bearer {token}"

        warnings.filterwarnings('ignore')

    def _get_file_list(self, sub_dir_pattern):
        if self.use_hf:
            path_pattern = f"datasets/{self.repo_id}/{sub_dir_pattern}"
            return sorted(self.fs.glob(path_pattern))
        else:
            return sorted(glob.glob(os.path.join(self.source_path, sub_dir_pattern)))

    def _get_readable_path(self, file_identifier):
        if self.use_hf:
            prefix = f"datasets/{self.repo_id}/"
            rel_path = file_identifier.replace(prefix, "")

            url = hf_hub_url(repo_id=self.repo_id, filename=rel_path, repo_type="dataset")

            if rel_path.endswith('.tif') or rel_path.endswith('.tiff'):
                return f"/vsicurl/{url}"
            return url
        else:
            return file_identifier

    def get_dataset(self, dataset, start_date=None, end_date=None):
        dataset_lower = dataset.lower()

        if dataset_lower == 'tahmo':
            ds = self._load_station()
        elif dataset_lower == 'nasadem':
            ds = self._load_static()
            return ds
        elif dataset_lower in ['imerg', 'chirps', 'era5', 'tamsat']:
            ds = self._load_grid(dataset_lower, start_date, end_date)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if 'time' in ds.dims and (start_date or end_date):
            if not np.issubdtype(ds.time.dtype, np.datetime64):
                ds['time'] = pd.to_datetime(ds.time.values)
            ds = ds.sel(time=slice(start_date, end_date))

        return ds

    def _load_grid(self, name, start_date=None, end_date=None):
        file_pattern = f"data/grids/{name}/{name}_*.tif"
        files = self._get_file_list(file_pattern)

        if not files:
            files = self._get_file_list(f"grids/{name}/{name}_*.tif")

        if not files:
            raise FileNotFoundError(f"No files found for {name}")

        if start_date or end_date:
            filtered_files = []
            start_dt = pd.to_datetime(start_date) if start_date else pd.Timestamp.min
            end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.max

            for f in files:
                try:
                    filename = os.path.basename(f)
                    date_part = os.path.splitext(filename)[0].split('_')[-1]
                    file_date = pd.to_datetime(date_part)
                    if start_dt.date() <= file_date.date() <= end_dt.date():
                        filtered_files.append(f)
                except:
                    continue
            files = filtered_files

        if not files:
            raise FileNotFoundError(f"No files found for {name} in range")

        datasets = []
        print(f"Streaming {len(files)} daily files for {name}...")

        for f in files:
            try:
                path = self._get_readable_path(f)
                da = rioxarray.open_rasterio(path, chunks='auto')

                timestamps = []
                if 'long_name' in da.attrs:
                    meta_val = da.attrs['long_name']
                    if isinstance(meta_val, str):
                        timestamps = [meta_val]
                    else:
                        timestamps = list(meta_val)

                if not timestamps:
                    filename = os.path.basename(f)
                    date_part = os.path.splitext(filename)[0].split('_')[-1]
                    if da.shape[0] == 1:
                        timestamps = [pd.to_datetime(date_part).isoformat()]
                    else:
                        continue

                times = pd.to_datetime(timestamps)
                da = da.assign_coords(band=times)
                da = da.rename({'band': 'time'})

                datasets.append(da)
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not datasets:
            raise ValueError("Failed to load any data files.")

        ds = xr.concat(datasets, dim='time')
        ds = ds.sortby('time')

        if 'x' in ds.dims and 'y' in ds.dims:
            ds = ds.rename({'x': 'lon', 'y': 'lat'})

        ds.name = 'precipitation'
        return ds.to_dataset()

    def _load_station(self):
        files = self._get_file_list("data/stations/tahmo_precipitation.csv")
        if not files: files = self._get_file_list("stations/tahmo_precipitation.csv")

        if not files: raise FileNotFoundError("TAHMO files not found")

        precip_url = self._get_readable_path(files[0])
        meta_url = precip_url.replace("tahmo_precipitation.csv", "tahmo_stations_topography.csv")

        # FIX: Add auth headers for Pandas
        storage_options = None
        if self.use_hf and self.token:
            storage_options = {"Authorization": f"Bearer {self.token}"}

        try:
            df = pd.read_csv(precip_url, index_col='time', parse_dates=True, storage_options=storage_options)
            meta_df = pd.read_csv(meta_url, storage_options=storage_options).set_index('station_id')
        except Exception as e:
            raise FileNotFoundError(f"Could not read TAHMO CSVs: {e}")

        da = xr.DataArray(df.values, dims=['time', 'station_id'], coords={'time': df.index, 'station_id': df.columns}, name='precipitation')
        ds = da.to_dataset()

        for col in meta_df.columns:
            ds.coords[col] = ('station_id', meta_df[col].reindex(ds.station_id.values))

        return ds

    def _load_static(self):
        files = self._get_file_list("data/static/nasadem_topography.tif")
        if not files: files = self._get_file_list("static/nasadem_topography.tif")

        if not files: raise FileNotFoundError("NASADEM file not found")

        path = self._get_readable_path(files[0])
        ds_raw = rioxarray.open_rasterio(path, chunks='auto')
        ds = ds_raw.to_dataset(dim='band')
        ds = ds.rename({1: 'elevation', 2: 'slope', 3: 'aspect'})

        if 'x' in ds.dims and 'y' in ds.dims:
            ds = ds.rename({'x': 'lon', 'y': 'lat'})

        return ds