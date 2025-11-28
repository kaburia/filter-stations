import requests
import pandas as pd
import dateutil.parser
import numpy as np
import json
import datetime
from tqdm.auto import tqdm
import multiprocessing as mp
from collections import Counter
import itertools

import warnings
warnings. filterwarnings('ignore')


# Constants
API_BASE_URL = 'https://datahub.tahmo.org'
API_MAX_PERIOD = '365D'

endpoints = {'VARIABLES': 'services/assets/v2/variables', # 28 different variables
             'STATION_INFO': 'services/assets/v2/stations',
             'WEATHER_DATA': 'services/measurements/v2/stations', # Configured before requesting
             'DATA_COMPLETE': 'custom/sensordx/latestmeasurements',
             'STATION_STATUS': 'custom/stations/status',
             'QUALITY_OBJECTS': 'custom/sensordx/reports'}

# module directory
# module_dir = os.path.dirname(__file__)

# Get data class
class RetrieveData:
    def __init__(self, apiKey, apiSecret):
        self.apiKey = apiKey
        self.apiSecret = apiSecret

    def __handleApiError(self, apiRequest):
        json =None
        try:
            json = apiRequest.json()
        finally:
            if json and 'error' in json and 'message' in json['error']:
                print(json)
                raise Exception(json['error']['message'])
            else:
                raise Exception(f'API request failed with status code {apiRequest.status_code}')

    def __request(self, endpoint, params):
        # print(f'API request: {endpoint}')
        apiRequest = requests.get(f'{API_BASE_URL}/{endpoint}',
                                    params=params,
                                    auth=requests.auth.HTTPBasicAuth(
                                    self.apiKey,
                                    self.apiSecret
                                )
        )
        if apiRequest.status_code == 200:
            return apiRequest.json()
        else:
            return self.__handleApiError(apiRequest)

    def get_stations_info(self, station=None, multipleStations=[], countrycode=None, list_coords=None):
        """
        Retrieves information about weather stations from an API endpoint and returns relevant information based on the parameters passed to it.

        Parameters:
        -----------
        - station (str, optional): Code for a single station to retrieve information for. Defaults to None.
        - multipleStations (list, optional): List of station codes to retrieve information for multiple stations. Defaults to [].
        - countrycode (str, optional): Country code to retrieve information for all stations located in the country. Defaults to None.
        - list_coords (list, optional): List of coordinates to filter stations within a certain bounding box. Defaults to None.

        Returns:
        -----------
        - pandas.DataFrame: DataFrame containing information about the requested weather stations.

        Usage:
        -----------
        To retrieve information about a single station:
        ```python
        station_info = ret.get_stations_info(station='TA00001')
        ```
        To retrieve information about multiple stations:
        ```python
        station_info = ret.get_stations_info(multipleStations=['TA00001', 'TA00002'])
        ```
        To retrieve information about all stations in a country:
        ```python
        station_info = ret.get_stations_info(countrycode='KE')
        ```

        """
        # Make API request and convert response to DataFrame
        response = self.__request(endpoints['STATION_INFO'], {'sort':'code'})
        info = pd.json_normalize(response['data']).drop('id', axis=1)

        # remove columns with TH in the code
        info = info.drop(labels=info['code'][info.code.str.contains('TH')].index, axis=0)

        # Filter DataFrame based on parameters
        if station:
            return info[info['code'] == station.upper()]
        elif len(multipleStations) >= 1:
            return info[info['code'].isin(multipleStations)]
        elif countrycode:
            info = info[info['location.countrycode'] == f'{countrycode.upper()}']
            return info.drop(labels=info['code'][info.code.str.contains('TH')].index, axis=0)
        elif list_coords:
            if len(list_coords) != 4:
                raise ValueError('Pass in a list of 4 coordinates [min_lon, min_lat, max_lon, max_lat]')
            min_lon, min_lat, max_lon, max_lat = list_coords
            return info[(info['location.longitude'] >= min_lon) &
                        (info['location.longitude'] <= max_lon) &
                        (info['location.latitude'] >= min_lat) &
                        (info['location.latitude'] <= max_lat)]
        else:
            return info

    # get station coordinates
    def get_coordinates(self, station_sensor, normalize=False):
        """
        Retrieve longitudes, latitudes for a list of station_sensor names.

        Parameters
        ----------
        station_sensor : list
            List of station_sensor names.
        normalize : bool
            If True, normalize the coordinates using MinMaxScaler to the range (0,1).

        Returns
        -------
        pd.DataFrame
            DataFrame containing longitude and latitude coordinates.

        Usage
        -----
        To retrieve coordinates::

            start_date = '2023-01-01'
            end_date = '2023-12-31'
            country= 'KE'

            # get the precipitation data
            ke_pr = filt.filter_pr(start_date=start_date, end_date=end_date, country='Kenya')
            
            # get the coordinates
            xs = ret.get_coordinates(ke_pr.columns, normalize=True)
        """

        station_sensor = sorted(station_sensor)
        # Extract unique station names
        unique_stations = set([station.split('_')[0] for station in station_sensor])

        # Get information for all stations in a single call
        all_stations_info = self.get_stations_info(multipleStations=list(unique_stations))

        # A dictionary with the number of stations per sensor
        stations_count = Counter([station.split('_')[0] for station in station_sensor])

        # coordinates list to store lat,lon
        coordinates_list = []

        # coord dict
        coord_dict = dict()

        # loop through the dictionary appending to the coordinates_list duplicating depending on the count
        for station, count in stations_count.items():
            # Extract latitude and longitude information for the station
            station_info = all_stations_info[all_stations_info.code == station][['location.longitude', 'location.latitude']].values[0]
            coordinates_list.extend([list(station_info)] * count)
            # append to the coord_dict
            coord_dict[station] = coordinates_list
            coordinates_list = []

        coordinates_array = np.vstack([np.array(coords) for coords in coord_dict.values()])

        # if normalize:
        #     # Normalize  (latitude and longitude)
        #     scaler = MinMaxScaler()
        #     coordinates_array = scaler.fit_transform(coordinates_array)

        if normalize:
            # Manual MinMax Scaling using NumPy
            min_val = np.min(coordinates_array, axis=0)
            max_val = np.max(coordinates_array, axis=0)
            # Avoid division by zero if max == min
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            
            coordinates_array = (coordinates_array - min_val) / range_val

        return pd.DataFrame(coordinates_array.T, columns=station_sensor, index=['longitude', 'latitude'])



    def get_variables(self):
        """
        Retrieves information about available weather variables from an API endpoint.

        Returns:
        -----------
        - dict: Dictionary containing information about available weather variables, keyed by variable shortcode.
        """
        # Make API request and create dictionary of variables
        response = self.__request(endpoints['VARIABLES'], {})
        variables = {}
        if 'data' in response and isinstance(response['data'], list):
            for element in response['data']:
                variables[element['variable']['shortcode']] = element['variable']
        return variables


     # Split date range into intervals of 365 days.
    def __splitDateRange(self, inputStartDate, inputEndDate):
        try:
            startDate = dateutil.parser.parse(inputStartDate)
            endDate = dateutil.parser.parse(inputEndDate)
        except ValueError:
            raise ValueError("Invalid data parameters")

        # Split date range into intervals of 365 days.
        dates = pd.date_range(start=startDate.strftime("%Y%m%d"), end=endDate.strftime("%Y%m%d"), freq=API_MAX_PERIOD)

        df = pd.DataFrame([[i, x] for i, x in
                           zip(dates, dates.shift(1) - datetime.timedelta(seconds=1))],
                          columns=['start', 'end'])

        # Set start and end date to their provided values.
        df.loc[0, 'start'] = pd.Timestamp(startDate)
        df['end'].iloc[-1] = pd.Timestamp(endDate)
        return df

    def k_neighbours(self, station, number=5):
        import haversine as hs

        """
        Returns a dictionary of the nearest neighbouring stations to the specified station.

        Parameters:
        -----------
        - station (str): Code for the station to find neighbouring stations for.
        - number (int, optional): Number of neighbouring stations to return. Defaults to 5.

        Returns:
        -----------
        - dict: Dictionary containing the station codes and distances of the nearest neighbouring stations.
        """
        # Get latitude and longitude of specified station
        lon, lat = self.get_stations_info(station)[['location.longitude', 'location.latitude']].values[0]

        # Calculate distances to all other stations and sort by distance
        infostations = self.get_stations_info()
        infostations['distance'] = infostations.apply(lambda row: hs.haversine((lat, lon), (row['location.latitude'], row['location.longitude'])), axis=1)
        infostations = infostations.sort_values('distance')

        # Create dictionary of nearest neighbouring stations and their distances
        return dict(infostations[['code', 'distance']].head(number).values[1:])

    # retrieve status of the stations
    def station_status(self):
        """
        Retrieves the status of all weather stations

        Returns:
        -----------
        - pandas.DataFrame: DataFrame containing the status of all weather stations.
        """
        # Make API request and convert response to DataFrame
        response = self.__request(endpoints['STATION_STATUS'], {})
        station_status = pd.json_normalize(response.json())
        station_status = station_status.drop(labels=station_status['id'][station_status.id.str.contains('TH')].index, axis=0)

        # create a column if active is true and offline_24h is false
        def active(row):
            if row['active'] == True and row['offline_24h'] == False:
                return 1
            else:
                return 0
        station_status['online'] = station_status.apply(active, axis=1)
        return station_status

    # get the qc flags
    # def qc_flags(self, station, startDate=None, endDate=None, variables=None):


    # trained models in stored in mongoDB
    def trained_models(self, columns=None):
        """
        Retrieves trained models from the MongoDB.

        Parameters:
        -----------
        - columns (list of str, optional): List of column names to include in the returned DataFrame.
                If None, all columns are included. Defaults to None.

        Returns:
        -----------
        - pandas.DataFrame: DataFrame containing trained models with the specified columns.
        """
        reqUrl = "https://sensordx.tahmo.org/api/models" # endpoint
        # response = self.__request(reqUrl, {})
        print(f'API request: {reqUrl}')
        apiRequest = requests.get(f'{reqUrl}',
                                    params={},
                                    auth=requests.auth.HTTPBasicAuth(
                                    self.apiKey,
                                    self.apiSecret
                                )
        )
        if apiRequest.status_code == 200:
            response =  apiRequest.json()
            # print(response)
            if columns:
                return pd.DataFrame(response)[columns]
            else:
                return pd.DataFrame(response)
        else:
            return self.__handleApiError(apiRequest)


    def aggregate_variables(self, dataframe, freq='1D', method='sum'):
        """
            Aggregates a pandas DataFrame of weather variables.

            Parameters
            ----------
            dataframe : pandas.DataFrame
                DataFrame containing weather variable data.
            freq : str, optional
                Frequency to aggregate by. Defaults to '1D'.
                Examples: '1H' (hourly), '1D' (daily), '1W' (weekly).
            method : str or callable, optional
                Method to use for aggregation. Defaults to 'sum'.
                Options: 'sum', 'mean', 'min', 'max'.
                
                Example of a custom method::

                    def custom_median(x):
                        return np.nan if x.isnull().all() else x.median()

                    data = aggregate_variables(df, freq='1D', method=custom_median)

            Returns
            -------
            pandas.DataFrame
                DataFrame containing aggregated weather variable data.

            Usage
            -----
            To aggregate data hourly::

                hourly_data = aggregate_variables(dataframe, freq='1H')

            To use a custom aggregation method::

                def custom_median(x):
                    return np.nan if x.isnull().all() else x.median()

                daily_median = aggregate_variables(df, freq='1D', method=custom_median)
        """


        # Define aggregation methods
        aggregation_methods = {
            'sum': lambda x: np.nan if x.isnull().all() else x.sum(),
            'mean': lambda x: np.nan if x.isnull().all() else x.mean(),
            'min': lambda x: np.nan if x.isnull().all() else x.min(),
            'max': lambda x: np.nan if x.isnull().all() else x.max()
        }

        # Determine the aggregation function to use
        if isinstance(method, str):
            if method not in aggregation_methods:
                raise ValueError('Invalid method. Method should be either "sum", "mean", "min", "max" or a custom function.')
            agg_func = aggregation_methods[method]
        elif callable(method):
            agg_func = lambda x: np.nan if x.isnull().all() else method(x)
        else:
            raise ValueError('Invalid method. Method should be either "sum", "mean", "min", "max" or a custom function.')

        # Convert 'Date' column to DateTimeIndex if it's not already
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            dataframe.index = pd.to_datetime(dataframe.index)

        # Groupby the existing time index
        aggregated_df = dataframe.groupby(pd.Grouper(freq=freq)).agg(agg_func)

        return aggregated_df

    # # aggregate qualityflags
    def aggregate_qualityflags(self, dataframe, freq='1D'):
        """
        Aggregate quality flags in a DataFrame by day.

        Parameters:
        -----------
        - dataframe (pd.DataFrame): The DataFrame containing the measurements.

        Returns:
        -----------
        - pd.DataFrame: A DataFrame with aggregated quality flags, where values greater than 1 are rounded up.

        """
        dataframe = dataframe.reset_index()
        dataframe.rename(columns={'index': 'Date'}, inplace=True)

        # Group by day and calculate the mean. If value that day is greater than 1, get the ceiling.
        # return dataframe.groupby(pd.Grouper(key='Date', axis=0, freq='1D')).mean().applymap(lambda x: ceil(x) if x > 1 else x)
        # return the max value for the day
        return dataframe.groupby(pd.Grouper(key='Date', axis=0, freq=freq)).max()
        # Function to count unique values for each column in a group
        # def count_unique_values(group):
        #     return {col: group[col].value_counts().to_dict() for col in group.columns if col != 'Date'}
        # return dataframe.groupby(pd.Grouper(key='Date', axis=0, freq=freq)).apply(count_unique_values)
    # def aggregate_qualityflags(self, dataframe, freq='1D'):
    #     """
    #     Aggregate quality flags in a DataFrame by a specified frequency.

    #     Parameters:
    #     -----------
    #     - dataframe (pd.DataFrame): The DataFrame containing the measurements.
    #     - freq (str): The frequency for aggregation (default is '1D' for one day).

    #     Returns:
    #     -----------
    #     - pd.DataFrame: A DataFrame with a new column 'aggregated_counts' containing dictionaries
    #                     of value counts for each aggregation period.
    #     """
    #     dataframe = dataframe.reset_index()
    #     dataframe.rename(columns={'index': 'Date'}, inplace=True)
    #     # Ensure 'Date' is a datetime column
    #     dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    #     # Group by the specified frequency
    #     grouped = dataframe.groupby(pd.Grouper(key='Date', axis=0, freq=freq))

    #     # Function to count unique values for each column in a group and return as a dictionary
    #     def count_unique_values(group):
    #         return {col: group[col].value_counts().to_dict() for col in group.columns if col != 'Date'}

    #     # Apply the function to each group and construct the result DataFrame
    #     result = grouped.apply(lambda group: pd.Series(count_unique_values(group))).reset_index()

    #     return result




    # Get the single stations data
     # Get the variables only
    def get_measurements(self, station, startDate=None, endDate=None, variables=None,
                     dataset='controlled', aggregate='5min',
                     quality_flags=False, quality_flags_filter=[1],
                         method='sum'):
      """
      Retrieve measurements from a station with fine control over time and quality.

      Parameters:
      - station (str): Station ID.
      - startDate (str): Start datetime (supports full ISO format).
      - endDate (str): End datetime (inclusive).
      - variables (list): List of variable codes to retrieve.
      - dataset (str): 'controlled' or 'raw'.
      - aggregate (str): Aggregation frequency (e.g., '5min', '30min').
      - quality_flags (bool): If True, return quality flags instead of values.
      - quality_flags_filter (list of int): Optional list of quality flag codes [1-4] to keep.

      Returns:
      - pd.DataFrame: Time-indexed data.
      """
      import gc
      import concurrent.futures

      # Parse full datetime strings
      startDate = pd.to_datetime(startDate, utc=True)
      endDate = pd.to_datetime(endDate, utc=True)

      # Cap end date to avoid incomplete intervals (align with aggregation)
      endDate += pd.Timedelta(minutes=-pd.Timedelta(aggregate).seconds // 60)

      endpoint = f'services/measurements/v2/stations/{station}/measurements/{dataset}'
      dateSplit = self.__splitDateRange(startDate.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                        endDate.strftime('%Y-%m-%dT%H:%M:%SZ'))

      series = []
      seriesHolder = {}

      # Helper function for parallel requests
      def fetch_chunk(row):
          params = {
              'start': row['start'].strftime('%Y-%m-%dT%H:%M:%SZ'),
              'end': row['end'].strftime('%Y-%m-%dT%H:%M:%SZ')
          }
          if variables and len(variables) == 1:
              params['variable'] = variables[0]
          
          try:
              return self.__request(endpoint, params)
          except Exception as e:
              print(f"Error fetching chunk {params['start']} to {params['end']}: {e}")
              return None

      # Execute requests in parallel
      with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
          futures = [executor.submit(fetch_chunk, row) for _, row in dateSplit.iterrows()]
          
          for future in concurrent.futures.as_completed(futures):
              response = future.result()
              if not response or not ('results' in response and response['results']):
                  continue

              for result in response['results']:
                  if 'series' not in result or not result['series']:
                      continue

                  for serie in result['series']:
                      columns = serie['columns']
                      observations = serie['values']
                      idx_map = {name: i for i, name in enumerate(columns)}
                      batch_vars = variables if variables else list(set([obs[idx_map['variable']] for obs in observations]))

                      for var in batch_vars:
                          filtered_obs = [obs for obs in observations if obs[idx_map['variable']] == var]
                          if var in seriesHolder:
                              seriesHolder[var].extend(filtered_obs)
                          else:
                              seriesHolder[var] = filtered_obs

      for var, obs_list in seriesHolder.items():
        time_idx = idx_map['time']
        val_idx = idx_map['value']
        qual_idx = idx_map['quality']
        sens_idx = idx_map['sensor']

        timestamps = [obs[time_idx] for obs in obs_list]
        duplicate_times = len(timestamps) > len(set(timestamps))

        if duplicate_times:
            sensors = set(obs[sens_idx] for obs in obs_list)
            for sensor in sensors:
                filtered = [obs for obs in obs_list if obs[sens_idx] == sensor]
                times = pd.to_datetime([obs[time_idx] for obs in filtered], utc=True)

                values = [obs[val_idx] if (not quality_flags_filter or obs[qual_idx] in quality_flags_filter) else np.nan for obs in filtered]
                flags = [obs[qual_idx] for obs in filtered]

                base_col = f"{var}_{station}_{sensor}" if len(seriesHolder) > 1 else f"{station}_{sensor}"

                value_series = pd.Series(values, index=pd.DatetimeIndex(times), dtype=np.float64)
                value_series = value_series[~value_series.index.duplicated(keep='first')]

                if quality_flags:
                    flag_series = pd.Series(flags, index=pd.DatetimeIndex(times), dtype=np.int32)
                    flag_series = flag_series[~flag_series.index.duplicated(keep='first')]
                    series.append(value_series.to_frame(base_col))
                    series.append(flag_series.to_frame(f"{base_col}_q_flag"))
                else:
                    series.append(value_series.to_frame(base_col))

        else:
            times = pd.to_datetime([obs[time_idx] for obs in obs_list], utc=True)

            values = [obs[val_idx] if (not quality_flags_filter or obs[qual_idx] in quality_flags_filter) else np.nan for obs in obs_list]
            flags = [obs[qual_idx] for obs in obs_list]

            base_col = f"{var}_{station}" if len(seriesHolder) > 1 else f"{station}"

            value_series = pd.Series(values, index=pd.DatetimeIndex(times), dtype=np.float64)
            value_series = value_series[~value_series.index.duplicated(keep='first')]

            if quality_flags:
                flag_series = pd.Series(flags, index=pd.DatetimeIndex(times), dtype=np.int32)
                flag_series = flag_series[~flag_series.index.duplicated(keep='first')]
                series.append(value_series.to_frame(base_col))
                series.append(flag_series.to_frame(f"{base_col}_q_flag"))
            else:
                series.append(value_series.to_frame(base_col))

        gc.collect()


      if series:
          df = pd.concat(series, axis=1, sort=False)
          df = df.sort_index()
          if quality_flags:
              return self.aggregate_qualityflags(df, freq=aggregate, method=method)
          return self.aggregate_variables(df, freq=aggregate, method=method)

      # If no data available
      idx = pd.date_range(start=startDate, end=endDate, freq=aggregate, tz='UTC')[:-1]
      return pd.DataFrame(index=idx, columns=[f'{station}'])

        # else:
              # drop any


            # # check if dataframe is empty
            # if len(series)
            #     print('No data found')
                # return df
            # else:
                # add the date range in the dataframe and the column as the station filled with NaN
                # df = pd.DataFrame(index=pd.date_range(start=startDate, end=endDate, tz='UTC', freq=aggregate), columns=[f'{station}'])
                # # remove the last row
                # return df[:-1]

            # else:
                # remove the last row
                # df = df # lacks values for the last day
                # if quality_flags:
                #     return self.aggregate_qualityflags(df, freq=aggregate)
                # return self.aggregate_variables(df, freq=aggregate)

    def multiple_measurements(self,
                              stations_list,
                              startDate,
                              endDate,
                              variables,
                              dataset='controlled',
                              csv_file=None,
                              aggregate='1D',
                              quality_flags=False,
                              num_workers=4):
        """
        Retrieves measurements for multiple stations within a specified date range.

        Parameters
        ----------
        stations_list : list
            A list of strings containing the codes of the stations to retrieve data from.
        startDate : str
            The start date for the measurements, in the format 'yyyy-mm-dd'.
        endDate : str
            The end date for the measurements, in the format 'yyyy-mm-dd'.
        variables : list
            A list of strings containing the names of the variables to retrieve.
        dataset : str, optional
            The name of the database to retrieve the data from. Default is 'controlled', alternatively 'raw'.
        csv_file : str, optional
            Pass the name of the csv file to save the data, otherwise it will return the dataframe.
        aggregate : str, optional
            Aggregation frequency. If '1D', aggregate per day.
        quality_flags : bool, optional
            If True, return quality flags instead of values.
        num_workers : int, optional
            Number of parallel workers. Defaults to 4.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the aggregated data for all stations.

        Raises
        ------
        ValueError
            If stations_list is not a list.

        Example
        -------
        To retrieve precipitation data for stations in Kenya for the last week and save it as a csv file::

            # Import the necessary modules
            from datetime import datetime, timedelta
            from filter_stations import RetrieveData

            # An instance of the RetrieveData class
            ret = RetrieveData(apiKey, apiSecret)

            # Get today's date
            today = datetime.now()
            last_week = today - timedelta(days=7)

            # Format date as a string
            last_week_str = last_week.strftime('%Y-%m-%d')
            today_str = today.strftime('%Y-%m-%d')

            # Define the list of stations
            stations = ['TA00001', 'TA00002']
            variables = ['pr']

            # Call the multiple_measurements method
            aggregated_data = ret.multiple_measurements(
                stations, last_week_str, today_str, variables,
                dataset='raw', csv_file='Kenya_precipitation_data', aggregate='1D'
            )
        """
        if not isinstance(stations_list, list):
            raise ValueError('Pass in a list')

        error_dict = {}
        # default to 4 workers/cores
        if num_workers is None or num_workers < 1:
            num_workers = 4
        # Create a multiprocessing pool with the specified number of workers
        pool = mp.Pool(processes=num_workers)

        try:
            results = []
            with tqdm(total=len(stations_list), desc='Retrieving data for stations') as pbar:
                for station in stations_list:
                    results.append(pool.apply_async(self.get_measurements, args=(station, startDate, endDate, variables, dataset, aggregate, quality_flags), callback=lambda _: pbar.update(1)))

                pool.close()
                pool.join()

            df_stats = [result.get() for result in results if isinstance(result.get(), pd.DataFrame)]

            if len(df_stats) > 0:
                df = pd.concat(df_stats, axis=1)
                if csv_file:
                    df.to_csv(f'{csv_file}.csv')
                    return df
                else:
                    return df
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            pool.terminate()

    # def get_variables_xarray(self, startDate, endDate, aggregate='5min', method='mean', num_workers=None):
    #   """
    #   Retrieves all variables from all stations and returns an xarray Dataset with time, station coordinates,
    #   and variables as data variables, along with station latitudes and longitudes.

    #   Parameters:
    #   -----------
    #   - startDate (str): Start date in 'YYYY-MM-DD' format.
    #   - endDate (str): End date in 'YYYY-MM-DD' format.
    #   - aggregate (str): Aggregation frequency (default '5min').
    #   - method (str): Aggregation method (default 'mean').

    #   Returns:
    #   -----------
    #   - xarray.Dataset: Dataset with dimensions (time, station), variables as data variables,
    #                     and station coordinates including latitude and longitude.
    #   """
    #   stations_df = self.get_stations_info(countrycode='RW')[['code', 'location.latitude', 'location.longitude']]
    #   stations = stations_df['code'].tolist()
    #   variables = list(self.get_variables().keys())

    #   def process_station(station):
    #       try:
    #           df = self.get_measurements(
    #               station, startDate, endDate, variables=variables,
    #               dataset='controlled', aggregate=aggregate, method=method
    #           )
    #           if df.empty:
    #               return None

    #           processed_data = {}
    #           for col in df.columns:
    #               if station not in col:
    #                   continue
    #               var = col.split('_')[0]
    #               processed_data.setdefault(var, []).append(df[col])

    #           if not processed_data:
    #               print(f"No matching variables for station {station} in columns: {df.columns.tolist()}")
    #               return None

    #           for var in processed_data:
    #               combined = pd.concat(processed_data[var], axis=1)
    #               processed_data[var] = combined.mean(axis=1)

    #           df_station = pd.DataFrame(processed_data)
    #           df_station['station'] = station
    #           df_station = df_station.reset_index().rename(columns={'index': 'time'})
    #           return df_station

    #       except Exception as e:
    #           print(f"Error processing station {station}: {str(e)}")
    #           return None

    #   # Optimized parallel processing
    #   def chunker(seq, size):
    #       return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    #   # Determine optimal chunk size
    #   num_workers = num_workers or mp.cpu_count()
    #   chunk_size = max(1, len(stations) // (num_workers * 2))

    #   chunks = list(chunker(stations, chunk_size))
    #   args_list = [
    #       (self.apiKey, self.apiSecret, chunk, variables, startDate, endDate, aggregate, method)
    #       for chunk in chunks
    #   ]

    #   with mp.Pool(processes=num_workers) as pool:
    #       results = []
    #       for result in tqdm(pool.imap_unordered(process_station_wrapper, args_list), total=len(chunks), desc='Processing station chunks'):
    #           results.extend(result)

    #   results = [r for r in results if r is not None]
    #   if not results:
    #       return xr.Dataset()

    #   combined_df = pd.concat(results, ignore_index=True)
    #   combined_df['time'] = pd.to_datetime(combined_df['time'])
    #   combined_long = combined_df.melt(id_vars=['time', 'station'], var_name='variable', value_name='value')
    #   combined_wide = combined_long.pivot_table(index=['time', 'station'], columns='variable', values='value')
    #   ds = combined_wide.to_xarray()

    #   # Attach coordinates
    #   stations_meta = stations_df.set_index('code')
    #   valid_stations = [s for s in ds.station.values if s in stations_meta.index]
    #   ds = ds.sel(station=valid_stations)
    #   ds.coords['latitude'] = ('station', stations_meta.loc[valid_stations, 'location.latitude'].values)
    #   ds.coords['longitude'] = ('station', stations_meta.loc[valid_stations, 'location.longitude'].values)

    #   # Attach variable metadata
    #   vars_list = self.get_variables()
    #   for var in ds.data_vars:
    #       if var in vars_list:
    #           ds[var].attrs = {
    #               'long_name': vars_list[var]['name'],
    #               'units': vars_list[var]['units'],
    #               'description': vars_list[var]['description']
    #           }

    #   ds.attrs.update({
    #       'title': 'TAHMO Weather Station Network Data',
    #       'processing_method': f'Aggregated with {num_workers} workers',
    #       'processing_date': datetime.datetime.utcnow().isoformat()
    #   })

    #   return ds

    def get_variables_xarray(self, startDate,
                             endDate,
                             variables=None,
                             stations_metadata=None,
                             aggregate='5min',
                             method='mean',
                             quality_flags_filter=[1]):
        import xarray as xr

        """
        Retrieves all variables from all stations and returns an xarray Dataset with time, station coordinates,
        and variables as data variables, along with station latitudes and longitudes.

        Parameters:
        -----------
        - startDate (str): Start date in 'YYYY-MM-DD' format.
        - endDate (str): End date in 'YYYY-MM-DD' format.
        - variables (list): A list of strings (shortcode) containing the names of the variables to retrieve
        - stations_metadata (pandas.DataFrame): DataFrame containing station metadata. (containing the station id and the coordinates)
        - aggregate (str): Aggregation frequency (default '1D' for daily).
        - method (str): Aggregation method (default 'mean').

        Returns:
        -----------
        - xarray.Dataset: Dataset with dimensions (time, station), variables as data variables,
                          and station coordinates including latitude and longitude.
        """
        # Get all stations metadata
        stations_df = self.get_stations_info(countrycode='KE')[['code', 'location.latitude', 'location.longitude']]
        stations = stations_df['code'].tolist()

        # list of variables
        vars_list = self.get_variables()
        variables = list(vars_list)

        # List to collect all data
        all_data = []

        # For each station, fetch and process data
        for station in stations:
            print(f"Processing station {station}")
            try:
                # Get measurements for all variables
                df = self.get_measurements(
                    station, startDate, endDate, variables=variables,
                    dataset='controlled', aggregate=aggregate, method=method,
                    quality_flags_filter=quality_flags_filter
                )
                # print(df)
                if df.empty:
                    continue

                # Process each variable to handle multiple sensors
                processed_data = {}
                # for col in df.columns:
                #   # Split using the unique separator
                #   parts = col.split('_')
                #   if len(parts) != 3:
                #       continue  # Skip malformed columns
                #   var, stn, sensor = parts
                #   if stn != station:
                #       continue
                #   if var not in processed_data:
                #       processed_data[var] = []
                #   processed_data[var].append(df[col])

                for col in df.columns:
                    # Extract variable name (assuming format 'var_station_sensor' or 'var_station')
                    parts = col.split('_')
                    var = parts[0]
                    # Check if the variable is from the current station
                    if parts[1] != station:
                        continue
                    # Aggregate sensor measurements for the same variable
                    if var not in processed_data:
                        processed_data[var] = []
                    processed_data[var].append(df[col])

                # Average across sensors for each variable
                for var in processed_data:
                    combined = pd.concat(processed_data[var], axis=1)
                    processed_data[var] = combined.mean(axis=1)

                # Create DataFrame for current station
                df_station = pd.DataFrame(processed_data)
                df_station['station'] = station
                df_station = df_station.reset_index().rename(columns={'index': 'time'})
                all_data.append(df_station)

            except Exception as e:
                print(f"Error processing station {station}: {e}")
                continue

        if not all_data:
            return xr.Dataset()

        # Combine all DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)

        # set the index time to string
        combined_df['time'] = combined_df['time'].astype(str)

        # Melt to long format and pivot to (time, station) index
        combined_long = combined_df.melt(id_vars=['time', 'station'], var_name='variable', value_name='value')
        combined_wide = combined_long.pivot_table(index=['time', 'station'], columns='variable', values='value')

        # Convert to xarray Dataset
        ds = combined_wide.to_xarray()

        # Add latitude/longitude coordinates FIRST
        stations_meta = stations_df.set_index('code')
        try:
            # Get valid stations that exist in both dataset and metadata
            valid_stations = [s for s in ds.station.values if s in stations_meta.index]

            ds = ds.sel(station=valid_stations)

            # Add latitude/longitude coordinates
            ds.coords['latitude'] = ('station', stations_meta.loc[valid_stations, 'location.latitude'].values)
            ds.coords['longitude'] = ('station', stations_meta.loc[valid_stations, 'location.longitude'].values)

            # Add the elevation
            ds.coords['elevation'] = ('station', stations_meta.loc[valid_stations, 'location.elevation'].values)


        except KeyError as e:
            print(f"Warning: Missing coordinates for some stations - {str(e)}")
            missing = list(set(ds.station.values) - set(stations_meta.index))
            print(f"Stations missing metadata: {missing}")
        # ds['latitude'] = xr.DataArray(
        #     stations_meta['location.latitude'].loc[ds.station.values].values,
        #     dims='station',
        #     coords={'station': ds.station}
        # )
        # ds['longitude'] = xr.DataArray(
        #     stations_meta['location.longitude'].loc[ds.station.values].values,
        #     dims='station',
        #     coords={'station': ds.station}
        # )

        # THEN add variable attributes
        for var in ds.data_vars:
            if var in vars_list:
                ds[var].attrs = {
                    'long_name': vars_list[var]['name'],
                    'units': vars_list[var]['units'],
                    'description': vars_list[var]['description']
                }

        # Add coordinate attributes
        ds['latitude'].attrs = {
            'long_name': 'Latitude',
            'units': 'degrees_north',
            'standard_name': 'latitude'
        }
        ds['longitude'].attrs = {
            'long_name': 'Longitude',
            'units': 'degrees_east',
            'standard_name': 'longitude'
        }

        ds['elevation'].attrs = {
            'long_name': 'Elevation',
            'units': 'm',
            'standard_name': 'elevation'
        }

        # Add global attributes
        ds.attrs = {
            'title': 'TAHMO Weather Station Network Data',
            'institution': 'TAHMO',
            'source': 'TAHMO API',
            'history': f'Created {datetime.datetime.now().isoformat()}',
            'Conventions': 'CF-1.8'
        }

        return ds

    # # multiple quality flags for multiple stations
    def multiple_qualityflags(self, stations_list, startDate, endDate, csv_file=None):
        """
        Retrieves and aggregates quality flag data for multiple stations.

        Parameters
        ----------
        stations_list : list
            A list of station codes.
        startDate : str
            The start date in 'YYYY-MM-DD' format.
        endDate : str
            The end date in 'YYYY-MM-DD' format.
        csv_file : str, optional
            The name of the CSV file to save the data. Default is None.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame containing the aggregated quality flag data, or None if an error occurs.

        Raises
        ------
        ValueError
            If stations_list is not a list.
        """
        error_dict = dict()

        if isinstance(stations_list, list):
            df_stats = []

            for station in stations_list:
                print(stations_list.index(station))
                print(f'Retrieving data for station: {station}')
                try:
                    data = self.get_measurements(station, startDate, endDate, variables=['pr'], quality_flags=True)
                    agg_data = self.aggregate_qualityflags(data)
                    df_stats.append(agg_data)
                except Exception as e:
                    error_dict[station] = f'{e}'

            with open("Errors.json", "w") as outfile:
                json.dump(error_dict, outfile, indent=4)

            if len(df_stats) > 0:
                df = pd.concat(df_stats, axis=1)
                df.to_csv(f'{csv_file}.csv')
                return df.reindex(sorted(df.columns),axis=1) #sorted dataframe


    def create_neighbor_graph(self, ds, threshold_km):
        """
        Creates a weighted graph of stations based on geographic proximity.

        Parameters:
        -----------
        ds (xarray.Dataset): Dataset containing station coordinates
        threshold_km (float): Maximum connection distance in kilometers

        Returns:
        -----------
        nx.Graph: NetworkX graph with:
            - Nodes: Station IDs with latitude/longitude attributes
            - Edges: Connections between stations within threshold distance
            - Edge weights: Haversine distances in kilometers
        """
        import networkx as nx
        import haversine as hs


        G = nx.Graph()

        # Add nodes with coordinates
        stations = ds.station.values
        variables = list(ds.data_vars)


        for station in stations:
            try:
                lat = ds.latitude.sel(station=station).item()
                lon = ds.longitude.sel(station=station).item()
                # Extract time-series data for each variable
                var_attrs = {}
                for var in variables:
                    try:
                        series = ds[var].sel(station=station).values
                        var_attrs[var] = series.tolist()  # Convert to native Python list
                    except KeyError:
                        var_attrs[var] = []  # Fallback if variable not available

                G.add_node(station, latitude=lat, longitude=lon, **var_attrs)
            except KeyError:
                print(f"Warning: Missing coordinates for station {station}, skipping")
                continue

        # Create all possible station pairs
        valid_stations = [n for n in G.nodes]
        pairs = list(itertools.combinations(valid_stations, 2))

        # Add edges for nearby stations
        for station1, station2 in pairs:
            coords1 = (G.nodes[station1]['latitude'], G.nodes[station1]['longitude'])
            coords2 = (G.nodes[station2]['latitude'], G.nodes[station2]['longitude'])

            distance = hs.haversine(coords1, coords2)

            if distance <= threshold_km:
                G.add_edge(station1, station2, weight=distance)

        return G

    def multiple_qualityflags(self, stations_list, startDate, endDate, csv_file=None):
        """
        Retrieves and aggregates quality flag data for multiple stations within a specified date range.

        Parameters
        ----------
        stations_list : list
            A list of station codes for which to retrieve data.
        startDate : str
            The start date in 'YYYY-MM-DD' format.
        endDate : str
            The end date in 'YYYY-MM-DD' format.
        csv_file : str, optional
            The name of the CSV file to save the aggregated data. Default is None.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame containing the aggregated quality flag data for the specified stations,
            or None if an error occurs.

        Raises
        ------
        ValueError
            If stations_list is not a list.
        """
        if not isinstance(stations_list, list):
            raise ValueError('Pass in a list')

        error_dict = {}
        pool = mp.Pool(processes=mp.cpu_count())  # Use all available CPU cores

        try:
            results = []
            with tqdm(total=len(stations_list), desc='Retrieving Quality Flags for stations') as pbar:
                for station in stations_list:
                    results.append(pool.apply_async(self.get_measurements, args=(station, startDate, endDate, ['pr'], 'controlled', True), callback=lambda _: pbar.update(1)))

                pool.close()
                pool.join()

            df_stats = []
            for result in results:
                try:
                    data = result.get()
                    agg_data = self.aggregate_qualityflags(data)
                    df_stats.append(agg_data)
                except Exception as e:
                    error_dict[station] = f'{e}'

            if len(df_stats) > 0:
                df = pd.concat(df_stats, axis=1)
                if csv_file:
                    df.to_csv(f'{csv_file}.csv')
                return df.reindex(sorted(df.columns), axis=1)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            pool.terminate()

        with open("Errors.json", "w") as outfile:
            json.dump(error_dict, outfile, indent=4)

        return None

def process_station_wrapper(args):
    apiKey, apiSecret, station_chunk, variables, startDate, endDate, aggregate, method = args
    local_rd = RetrieveData(apiKey, apiSecret, "")
    chunk_results = []

    for station in station_chunk:
        try:
            df = local_rd.get_measurements(
                station, startDate, endDate, variables=variables,
                dataset='controlled', aggregate=aggregate, method=method
            )
            if df.empty:
                continue

            processed_data = {}
            for col in df.columns:
                parts = col.split('_')
                var = parts[0]
                if parts[1] != station:
                    continue
                processed_data.setdefault(var, []).append(df[col])

            for var in processed_data:
                combined = pd.concat(processed_data[var], axis=1)
                processed_data[var] = combined.mean(axis=1)

            df_station = pd.DataFrame(processed_data)
            df_station['station'] = station
            df_station = df_station.reset_index().rename(columns={'index': 'time'})
            chunk_results.append(df_station)

        except Exception as e:
            print(f"Error processing {station}: {str(e)}")
            continue

    return chunk_results
