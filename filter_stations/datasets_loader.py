# import xarray as xr
# import pandas as pd
# import numpy as np
# import os
# import glob
# import rioxarray
# import warnings
# from huggingface_hub import HfFileSystem, hf_hub_url

# class RainLoader:
#     """
#     A unified loader for streaming climate data from Hugging Face or local storage.
#     Supports gridded satellite/reanalysis data (IMERG, CHIRPS, ERA5, TAMSAT),
#     station data (TAHMO), and static topography (NASADEM).
#     """
#     def __init__(self, source_path='', use_hf=True, repo_id='kaburia/DSAIL-weather-data', token=None):
#         self.use_hf = use_hf
#         self.repo_id = repo_id
#         self.token = token
#         self.source_path = source_path

#         if use_hf:
#             if not repo_id:
#                 raise ValueError("repo_id is required when use_hf=True")
#             self.fs = HfFileSystem(token=token)

#             if token:
#                 os.environ['GDAL_HTTP_HEADERS'] = f"Authorization: Bearer {token}"

#         warnings.filterwarnings('ignore')

#     def _get_file_list(self, sub_dir_pattern):
#         if self.use_hf:
#             path_pattern = f"datasets/{self.repo_id}/{sub_dir_pattern}"
#             return sorted(self.fs.glob(path_pattern))
#         else:
#             return sorted(glob.glob(os.path.join(self.source_path, sub_dir_pattern)))

#     def _get_readable_path(self, file_identifier):
#         if self.use_hf:
#             prefix = f"datasets/{self.repo_id}/"
#             rel_path = file_identifier.replace(prefix, "")

#             url = hf_hub_url(repo_id=self.repo_id, filename=rel_path, repo_type="dataset")

#             if rel_path.endswith('.tif') or rel_path.endswith('.tiff'):
#                 return f"/vsicurl/{url}"
#             return url
#         else:
#             return file_identifier

#     def get_dataset(self, dataset, start_date=None, end_date=None):
#         """
#         Main entry point to retrieve climate datasets.

#         This method handles the logic for selecting the correct loader (Grid, Station, or Static)
#         and applying temporal filtering if applicable.

#         Parameters
#         ----------
#         dataset : str
#             Name of the dataset to retrieve. Case-insensitive.
#             Supported options:
            
#             * **Gridded:** 'IMERG', 'CHIRPS', 'ERA5', 'TAMSAT'
#             * **Stations:** 'TAHMO'
#             * **Static:** 'NASADEM'
            
#         start_date : str, optional
#             Start date for filtering in 'YYYY-MM-DD' format. 
#             Applicable only to time-series datasets (Grids and Stations).
#         end_date : str, optional
#             End date for filtering in 'YYYY-MM-DD' format.
#             Applicable only to time-series datasets.

#         Returns
#         -------
#         xarray.Dataset
#             The requested dataset with standardized coordinates:
            
#             * **Time:** 'time' (datetime64)
#             * **Space:** 'lat', 'lon' (WGS84)
#             * **Variables:** 'precipitation' (for rainfall data) or 'elevation'/'slope'/'aspect' (for NASADEM).


#         Example
#         -------
#         To retrieve IMERG satellite rainfall data::

#             loader = RainLoader(token="hf_...")
#             ds = loader.get_dataset('IMERG', '2024-01-01', '2024-01-31')

#         To retrieve TAHMO station data with the slope and aspect added to the metadata::

#             ds_stations = loader.get_dataset('TAHMO', '2024-01-01', '2024-01-31')
#         """
#         dataset_lower = dataset.lower()

#         if dataset_lower == 'tahmo':
#             ds = self._load_station()
#         elif dataset_lower == 'nasadem':
#             ds = self._load_static()
#             return ds
#         elif dataset_lower in ['imerg', 'chirps', 'era5', 'tamsat']:
#             ds = self._load_grid(dataset_lower, start_date, end_date)
#         else:
#             raise ValueError(f"Unknown dataset: {dataset}")

#         if 'time' in ds.dims and (start_date or end_date):
#             if not np.issubdtype(ds.time.dtype, np.datetime64):
#                 ds['time'] = pd.to_datetime(ds.time.values)
#             ds = ds.sel(time=slice(start_date, end_date))

#         return ds

#     def _load_grid(self, name, start_date=None, end_date=None):
#         file_pattern = f"data/grids/{name}/{name}_*.tif"
#         files = self._get_file_list(file_pattern)

#         if not files:
#             files = self._get_file_list(f"grids/{name}/{name}_*.tif")

#         if not files:
#             raise FileNotFoundError(f"No files found for {name}")

#         if start_date or end_date:
#             filtered_files = []
#             start_dt = pd.to_datetime(start_date) if start_date else pd.Timestamp.min
#             end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.max

#             for f in files:
#                 try:
#                     filename = os.path.basename(f)
#                     date_part = os.path.splitext(filename)[0].split('_')[-1]
#                     file_date = pd.to_datetime(date_part)
#                     if start_dt.date() <= file_date.date() <= end_dt.date():
#                         filtered_files.append(f)
#                 except:
#                     continue
#             files = filtered_files

#         if not files:
#             raise FileNotFoundError(f"No files found for {name} in range")

#         datasets = []
#         print(f"Streaming {len(files)} daily files for {name}...")

#         for f in files:
#             try:
#                 path = self._get_readable_path(f)
#                 da = rioxarray.open_rasterio(path, chunks='auto')

#                 timestamps = []
#                 if 'long_name' in da.attrs:
#                     meta_val = da.attrs['long_name']
#                     if isinstance(meta_val, str):
#                         timestamps = [meta_val]
#                     else:
#                         timestamps = list(meta_val)

#                 if not timestamps:
#                     filename = os.path.basename(f)
#                     date_part = os.path.splitext(filename)[0].split('_')[-1]
#                     if da.shape[0] == 1:
#                         timestamps = [pd.to_datetime(date_part).isoformat()]
#                     else:
#                         continue

#                 times = pd.to_datetime(timestamps)
#                 da = da.assign_coords(band=times)
#                 da = da.rename({'band': 'time'})

#                 datasets.append(da)
#             except Exception as e:
#                 print(f"Error loading {f}: {e}")

#         if not datasets:
#             raise ValueError("Failed to load any data files.")

#         ds = xr.concat(datasets, dim='time')
#         ds = ds.sortby('time')

#         if 'x' in ds.dims and 'y' in ds.dims:
#             ds = ds.rename({'x': 'lon', 'y': 'lat'})

#         ds.name = 'precipitation'
#         return ds.to_dataset()

#     def _load_station(self):
#         files = self._get_file_list("data/stations/tahmo_precipitation.csv")
#         if not files: files = self._get_file_list("stations/tahmo_precipitation.csv")

#         if not files: raise FileNotFoundError("TAHMO files not found")

#         precip_url = self._get_readable_path(files[0])
#         meta_url = precip_url.replace("tahmo_precipitation.csv", "tahmo_stations_topography.csv")

#         # FIX: Add auth headers for Pandas
#         storage_options = None
#         if self.use_hf and self.token:
#             storage_options = {"Authorization": f"Bearer {self.token}"}

#         try:
#             df = pd.read_csv(precip_url, index_col='time', parse_dates=True, storage_options=storage_options)
#             meta_df = pd.read_csv(meta_url, storage_options=storage_options).set_index('station_id')
#         except Exception as e:
#             raise FileNotFoundError(f"Could not read TAHMO CSVs: {e}")

#         da = xr.DataArray(df.values, dims=['time', 'station_id'], coords={'time': df.index, 'station_id': df.columns}, name='precipitation')
#         ds = da.to_dataset()

#         for col in meta_df.columns:
#             ds.coords[col] = ('station_id', meta_df[col].reindex(ds.station_id.values))

#         return ds

#     def _load_static(self):
#         files = self._get_file_list("data/static/nasadem_topography.tif")
#         if not files: files = self._get_file_list("static/nasadem_topography.tif")

#         if not files: raise FileNotFoundError("NASADEM file not found")

#         path = self._get_readable_path(files[0])
#         ds_raw = rioxarray.open_rasterio(path, chunks='auto')
#         ds = ds_raw.to_dataset(dim='band')
#         ds = ds.rename({1: 'elevation', 2: 'slope', 3: 'aspect'})

#         if 'x' in ds.dims and 'y' in ds.dims:
#             ds = ds.rename({'x': 'lon', 'y': 'lat'})

#         return ds

import xarray as xr
import pandas as pd
import numpy as np
import os
import warnings
import zarr
from huggingface_hub import HfFileSystem, hf_hub_download

class RainLoader:
    """
    A unified loader for Climate Data from Hugging Face.
    
    STRATEGY: 'Cache-On-Demand'
    - Downloads only the specific years requested.
    - Keeps them ZIPPED on disk to save space (20GB total vs 100GB extracted).
    - Opens them locally for robust, instant access.
    """
    def __init__(self, repo_id='DeKUT-DSAIL/weather-data', token=None):
        self.repo_id = repo_id
        self.token = token
        self.fs = HfFileSystem(token=token)
        self.repo_fs_root = f"datasets/{repo_id}"
        warnings.filterwarnings('ignore')

    def get_dataset(self, dataset, start_date=None, end_date=None):
        dataset_lower = dataset.lower()
        
        # 1. Dispatcher
        if dataset_lower == 'tahmo':
            ds = self._load_single_zarr("data/obs/tahmo.zarr.zip")
        elif dataset_lower in ['topography', 'nasadem']:
            ds = self._load_static_nc("data/topography/east_africa_static_priors.nc")
        elif dataset_lower in ['imerg', 'chirps', 'era5', 'tamsat']:
            ds = self._load_gridded_multi_year(dataset_lower, start_date, end_date)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # 2. Post-load Temporal Filtering
        if 'time' in ds.dims and (start_date or end_date):
            if not np.issubdtype(ds.time.dtype, np.datetime64):
                 try:
                     ds['time'] = pd.to_datetime(ds.time.values)
                 except:
                     pass
            ds = ds.sel(time=slice(start_date, end_date))

        return ds

    def _find_repo_path(self, relative_pattern):
        """Locates the file path inside the repo (checks root vs data/ folder)."""
        # 1. Check Root
        path_root = relative_pattern
        if self.fs.glob(f"{self.repo_fs_root}/{path_root}"):
            return path_root
        # 2. Check data/ subfolder
        path_data = f"data/{relative_pattern}"
        if self.fs.glob(f"{self.repo_fs_root}/{path_data}"):
            return path_data
        return None

    def _download_and_open(self, relative_path):
        """
        Downloads the ZIPPED file to local cache and reads it directly.
        This keeps the data compressed on disk.
        """
        print(f"Caching {os.path.basename(relative_path)}...")
        
        # Downloads to ~/.cache/huggingface/hub/...
        # This handles auth, caching, and integrity checks automatically
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=relative_path,
            repo_type="dataset",
            token=self.token
        )
        
        # Open the local zip file using fsspec's zip protocol
        # Try Root first (based on your file listing, this should work)
        try:
            return xr.open_zarr(f"zip::{local_path}", consolidated=False)
        except Exception:
            # Fallback: If it fails, scan for a nested folder (e.g. 2020.zarr/)
            # Scanning a local zip is instant (unlike remote)
            print("   ...Root open failed, scanning for internal group...")
            import zipfile
            with zipfile.ZipFile(local_path) as z:
                for name in z.namelist():
                    if name.endswith('.zgroup'):
                        group = os.path.dirname(name)
                        # Avoid empty string if it's at root
                        if group: 
                            return xr.open_zarr(f"zip::{local_path}", group=group, consolidated=False)
            
            # If all else fails
            raise RuntimeError(f"Could not find valid Zarr group in {local_path}")

    def _load_gridded_multi_year(self, name, start_date, end_date):
        # 1. Scan for available files using FS (fast metadata only)
        glob_pattern = f"{self.repo_fs_root}/data/{name}/*.zarr.zip"
        all_files = self.fs.glob(glob_pattern)
        
        if not all_files:
            glob_pattern = f"{self.repo_fs_root}/{name}/*.zarr.zip"
            all_files = self.fs.glob(glob_pattern)
            
        if not all_files:
            raise FileNotFoundError(f"No Zarr files found for {name}")

        # 2. Filter files by Requested Year
        files_to_load = []
        s_year = pd.to_datetime(start_date).year if start_date else 1900
        e_year = pd.to_datetime(end_date).year if end_date else 2100

        print(f"Scanning {len(all_files)} files for {name} ({s_year}-{e_year})...")

        for file_path in sorted(all_files):
            filename = os.path.basename(file_path)
            try:
                # Extract year (e.g. 2020.zarr.zip -> 2020)
                file_year = int(filename.split('.')[0])
                if s_year <= file_year <= e_year:
                    # We need the relative path for hf_hub_download
                    # Remove "datasets/RepoId/" prefix
                    rel_path = file_path.replace(f"{self.repo_fs_root}/", "")
                    files_to_load.append(rel_path)
            except ValueError:
                continue 

        if not files_to_load:
            raise ValueError(f"No files found for {name} in range {s_year}-{e_year}")

        # 3. Download and Open
        datasets = []
        for rel_path in files_to_load:
            try:
                ds = self._download_and_open(rel_path)
                datasets.append(ds)
            except Exception as e:
                print(f"Failed to load {rel_path}: {e}")

        if not datasets:
            raise RuntimeError("Could not open any valid Zarr files.")

        # 4. Concatenate
        ds_combined = xr.concat(datasets, dim='time', coords='minimal', compat='override')
        ds_combined = ds_combined.sortby('time')
        
        return ds_combined

    def _load_single_zarr(self, relative_pattern):
        rel_path = self._find_repo_path(relative_pattern)
        if not rel_path:
             raise FileNotFoundError(f"Could not find {relative_pattern}")
        return self._download_and_open(rel_path)

    def _load_static_nc(self, relative_pattern):
        rel_path = self._find_repo_path(relative_pattern)
        if not rel_path:
             raise FileNotFoundError(f"Could not find {relative_pattern}")
        
        print(f"Caching {os.path.basename(rel_path)}...")
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=rel_path,
            repo_type="dataset",
            token=self.token
        )
        
        # Use standard Xarray open (handles local NetCDF robustly)
        ds = xr.open_dataset(local_path, chunks='auto')
        
        if 'band' in ds.dims:
             var_names = list(ds.data_vars)
             if len(var_names) == 1:
                 ds = ds.rename({var_names[0]: 'elevation'})
                 
        return ds