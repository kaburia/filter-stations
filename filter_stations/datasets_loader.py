import xarray as xr
import pandas as pd
import numpy as np
import os
import warnings
import zarr
from huggingface_hub import HfFileSystem, hf_hub_download

class RainLoader:
    """

    Examples
    --------
    >>> from filter_stations import RainLoader
    >>> read_token = '' # Request dsail-info@dkut.ac.ke to get a token to access the data
    >>> loader = RainLoader(token=read_token)
    """
    def __init__(self, repo_id='DeKUT-DSAIL/weather-data', token=None):
        self.repo_id = repo_id
        self.token = token
        self.fs = HfFileSystem(token=token)
        self.repo_fs_root = f"datasets/{repo_id}"
        warnings.filterwarnings('ignore')
    
    def __init__(self, auth_session, repo_id='DeKUT-DSAIL/weather-data'):
        warnings.filterwarnings('ignore')
        self.repo_id = repo_id
        
        # 1. Ask the Gatekeeper for the Hugging Face token
        credentials = auth_session.request_access(
            service_name='huggingface', 
            details={
                "action": "initialize_rain_loader", 
                "repo_id": repo_id
            }
        )
        
        # 2. Extract the token and initialize local Hugging Face operations
        self.token = credentials.get('token')
        
        if not self.token:
            raise ValueError("Failed to retrieve the Hugging Face token from the gatekeeper.")
            
        # 3. Initialize the file system natively on the client's machine
        self.fs = HfFileSystem(token=self.token)
        self.repo_fs_root = f"datasets/{self.repo_id}"
        
        # print(f"Hugging Face connection successfully established for: {self.repo_id}")

    def get_dataset(self, dataset, start_date=None, end_date=None):
        """
        Main entry point to retrieve climate datasets (Gridded, Station, or Static).

        Parameters
        ----------
        dataset : str
            Name of the dataset. Options include:
            - Gridded: 'imerg', 'chirps', 'era5', 'tamsat'
            - Station: 'tahmo'
            - Static: 'topography', 'nasadem'
        start_date : str, optional
            Start date (YYYY-MM-DD). Required for time-series datasets.
        end_date : str, optional
            End date (YYYY-MM-DD). Required for time-series datasets.

        Returns
        -------
        xarray.Dataset
            The requested dataset.

        Examples
        --------
        >>> # User gets TAHMO (Stations)
        >>> ds_stations = loader.get_dataset('TAHMO', '2024-01-01', '2024-01-30')

        >>> # User gets IMERG (Grids) - Exact same interface
        >>> ds_tamsat = loader.get_dataset('tamsat', '2024-01-01', '2024-01-20')
        >>> ds_era5 = loader.get_dataset('era5', '2024-01-01', '2024-01-20')
        >>> ds_imerg = loader.get_dataset('imerg', '2024-01-01', '2024-01-20')
        >>> ds_chirps = loader.get_dataset('chirps', '2024-01-01', '2024-01-20')

        >>> # User gets Topography (Static)
        >>> ds_topo = loader.get_dataset('topography')
        """
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