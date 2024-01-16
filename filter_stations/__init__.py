"""
Installation
------------
To install the package, run the following command in your terminal:
```bash
pip install -U filter-stations
```
Getting Started
---------------
All methods require an API key and secret, which can be obtained by contacting TAHMO. <br>
- The ```retreive_data``` class is used to retrieve data from the TAHMO API endpoints.<br> 
- The ```Filter``` class is used to filter weather stations data based on things like distance and region.<br>
- The ```pipeline``` class is used to create a pipeline of filters to apply to weather stations based on how they correlate with water level data.<br>
- The ```Interactive_maps``` class is used to plot weather stations on an interactive map.<br>

```python
# Import the necessary modules
from filter_stations import retreive_data, Filter, pipeline, Interactive_maps

# Define the API key and secret
apiKey = 'your_api_key' # request from TAHMO
apiSecret = 'your_api_secret' # request from TAHMO
maps_key = 'your_google_maps_key' # retrieve from google maps platform

# Initialize the class
ret = retreive_data(apiKey, apiSecret, maps_key)
fs = Filter(apiKey, apiSecret, maps_key)
pipe = pipeline(apiKey, apiSecret, maps_key)
maps = Interactive_maps(apiKey, apiSecret, maps_key)
```


"""
import requests
from urllib.parse import quote
import pandas as pd
import argparse
import dateutil.parser
import math
import haversine as hs
import folium
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from io import BytesIO
import base64
import json
from folium.plugins import MeasureControl
import os
import datetime
import gc
from math import ceil
import statsmodels.api as sm
from matplotlib.dates import DateFormatter
from tqdm.auto import tqdm
import multiprocessing as mp
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import Point
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

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
module_dir = os.path.dirname(__file__)

# Get data class
class retreive_data:
    def __init__(self, apiKey, apiSecret, api_key):
        self.apiKey = apiKey
        self.apiSecret = apiSecret
        self.api_key = api_key

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
    
    def get_stations_info(self, station=None, multipleStations=[], countrycode=None):
        """
        Retrieves information about weather stations from an API endpoint and returns relevant information based on the parameters passed to it.

        Parameters:
        -----------
        - station (str, optional): Code for a single station to retrieve information for. Defaults to None.
        - multipleStations (list, optional): List of station codes to retrieve information for multiple stations. Defaults to [].
        - countrycode (str, optional): Country code to retrieve information for all stations located in the country. Defaults to None.

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
        else:
            return info
    
    # get station coordinates
    def get_coordinates(self, station_sensor, normalize=False):
        """
        Retrieve longitudes,latitudes for a list of station_sensor names and duplicated for stations with multiple sensors.

        Parameters:
        -----------
        - station_sensor (list): List of station_sensor names.
        - normalize (bool): If True, normalize the coordinates using MinMaxScaler to the range (0,1).

        Returns:
        -----------
        - pd.DataFrame: DataFrame containing longitude and latitude coordinates for each station_sensor.

        Usage:
        -----------
        To retrieve coordinates 
        ```python
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        country= 'KE'
        
        # get the precipitation data for the stations
        ke_pr = filt.filter_pr(start_date=start_date, end_date=end_date, 
                                country='Kenya').set_index('Date')
        
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

        if normalize:
            # Normalize  (latitude and longitude)
            scaler = MinMaxScaler()
            coordinates_array = scaler.fit_transform(coordinates_array)

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
        
    
    def aggregate_variables(self, dataframe, freq='1D'):
        """
        Aggregates a pandas DataFrame of weather variables by summing values across each day.

        Parameters:
        -----------
        - dataframe (pandas.DataFrame): DataFrame containing weather variable data.
        - freq (str, optional): Frequency to aggregate the data by. Defaults to '1D'.

        Returns:
        -----------
        - pandas.DataFrame: DataFrame containing aggregated weather variable data, summed by day.
        
        Usage:
        -----------
        Define the DataFrame containing the weather variable data:
        ```python
        dataframe = ret.get_measurements('TA00001', '2020-01-01', '2020-01-31', ['pr']) # data comes in 5 minute interval
        ```
        To aggregate data hourly:
        ```python
        hourly_data = ret.aggregate_variables(dataframe, freq='1H')
        ```
        To aggregate data by 12 hours:
        ```python
        half_day_data = ret.aggregate_variables(dataframe, freq='12H')
        ```
        To aggregate data by day:
        ```python
        daily_data = ret.aggregate_variables(dataframe, freq='1D')
        ```
        To aggregate data by week:
        ```python
        weekly_data = ret.aggregate_variables(dataframe, freq='1W')
        ```
        To aggregate data by month:
        ```python
        monthly_data = ret.aggregate_variables(dataframe, freq='1M')
        ```
        """
        dataframe = dataframe.reset_index()
        dataframe.rename(columns={'index':'Date'}, inplace=True)
        # check if the column is all nan
        if dataframe.iloc[:, 1].isnull().all():
                return dataframe.groupby(pd.Grouper(key='Date', axis=0, 
                                            freq=freq)).agg({f'{dataframe.columns[1]}': 
                                                             lambda x: np.nan if x.isnull().all() 
                                                             else x.isnull().sum()})    
        else:
                return dataframe.groupby(pd.Grouper(key='Date', axis=0, 
                                            freq=freq)).sum()
    
    # aggregate qualityflags
    def aggregate_qualityflags(self, dataframe):
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
        return dataframe.groupby(pd.Grouper(key='Date', axis=0, freq='1D')).mean().applymap(lambda x: ceil(x) if x > 1 else x)



    
    # Get the variables only
    def get_measurements(self, station, startDate=None, endDate=None, variables=None, dataset='controlled', aggregate='5min', quality_flags=False):
            """
                Get measurements from a station.

                Parameters:
                -----------
                - station (str): The station ID.
                - startDate (str, optional): The start date of the measurement period in the format 'YYYY-MM-DD'.
                - endDate (str, optional): The end date of the measurement period in the format 'YYYY-MM-DD'.
                - variables (list, optional): The variables to retrieve measurements for. If None, all variables are retrieved.
                - dataset (str, optional): The dataset to retrieve measurements from. Default is 'controlled'.
                - aggregate (bool, optional): Whether to aggregate the measurements by variable. Default is False.
                - quality_flags (bool, optional): Whether to include quality flag data. Default is False.

                Returns:
                -----------
                - A DataFrame containing the measurements.

                Usage:
                -----------
                To retrieve precipitation data for a station for the last month:
                ```python
                from datetime import datetime, timedelta

                # Get today's date
                today = datetime.now()

                # Calculate one month ago
                last_month = today - timedelta(days=30)

                # Format date as a string
                last_month_str = last_month.strftime('%Y-%m-%d')
                today_str = today.strftime('%Y-%m-%d')

                # Define the station you want to retrieve data from
                station = 'TA00001'
                variables = ['pr']
                dataset = 'raw'
                
                # aggregate the data to 30 minutes interval
                aggregate = '30min'

                # Call the get_measurements method to retrieve and aggregate data
                TA00001_data = ret.get_measurements(station, last_month_str, 
                                                    today_str, variables, 
                                                    dataset, aggregate)
                ```                

            """         
            #print('Get measurements', station, startDate, endDate, variables)
            endpoint = 'services/measurements/v2/stations/%s/measurements/%s' % (station, dataset)

            dateSplit = self.__splitDateRange(startDate, endDate)
            series = []
            seriesHolder = {}

            for index, row in dateSplit.iterrows():
                params = {'start': row['start'].strftime('%Y-%m-%dT%H:%M:%SZ'), 'end': row['end'].strftime('%Y-%m-%dT%H:%M:%SZ')}
                if variables and isinstance(variables, list) and len(variables) == 1:
                    params['variable'] = variables[0]
                response = self.__request(endpoint, params)
                if 'results' in response and len(response['results']) >= 1 and 'series' in response['results'][0] and len(
                    response['results'][0]['series']) >= 1 and 'values' in response['results'][0]['series'][0]:

                    for result in response['results']:
                        if 'series' in result and len(result['series']) >= 1 and 'values' in result['series'][0]:
                            for serie in result['series']:

                                columns = serie['columns']
                                observations = serie['values']

                                time_index = columns.index('time')
                                quality_index = columns.index('quality')
                                variable_index = columns.index('variable')
                                sensor_index = columns.index('sensor')
                                value_index = columns.index('value')

                                # Create list of unique variables within the retrieved observations.
                                if not isinstance(variables, list) or len(variables) == 0:
                                    shortcodes = list(set(list(map(lambda x: x[variable_index], observations))))
                                else:
                                    shortcodes = variables

                                for shortcode in shortcodes:

                                    # Create list of timeserie elements for this variable with predefined format [time, value, sensor, quality].
                                    timeserie = list(map(lambda x: [x[time_index], x[value_index] if x[quality_index] == 1 else np.nan, x[sensor_index], x[quality_index]],
                                                        list(filter(lambda x: x[variable_index] == shortcode, observations))))

                                    if shortcode in seriesHolder:
                                        seriesHolder[shortcode] = seriesHolder[shortcode] + timeserie
                                    else:
                                        seriesHolder[shortcode] = timeserie

                                    # Clean up scope.
                                    del timeserie

                                # Clean up scope.
                                del columns
                                del observations
                                del shortcodes

                    # Clean up scope and free memory.
                    del response
                    gc.collect()

            for shortcode in seriesHolder:
                # Check if there are duplicate entries in this timeseries (multiple sensors for same variable).
                timestamps = list(map(lambda x: x[0], seriesHolder[shortcode]))

                if len(timestamps) > len(set(timestamps)):
                    # Split observations per sensor.
                    print('Split observations for %s per sensor' % shortcode)
                    sensors = list(set(list(map(lambda x: x[2], seriesHolder[shortcode]))))
                    for sensor in sensors:
                        sensorSerie = list(filter(lambda x: x[2] == sensor, seriesHolder[shortcode]))
                        timestamps = list(map(lambda x: pd.Timestamp(x[0]), sensorSerie))
                        values = list(map(lambda x: x[1], sensorSerie))
                        serie = pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=np.float64)
                        if quality_flags:
                            q_flag = list(map(lambda x: x[3], sensorSerie))
                            serie = pd.Series(q_flag, index=pd.DatetimeIndex(timestamps), dtype=np.int32)
                            series.append(serie.to_frame('%s_%s_%s' % (station, sensor, 'Q_FLAG')))
                            continue
                        if len(variables)==1:
                            series.append(serie.to_frame('%s_%s' % (station, sensor)))
                        else:
                            series.append(serie.to_frame('%s_%s_%s' % (shortcode, station, sensor)))

                        # Clean up scope.
                        del sensorSerie
                        del timestamps
                        del values
                        del serie
                else:
                    # print(pd.DataFrame(seriesHolder[shortcode]))
                    values = list(map(lambda x: x[1], seriesHolder[shortcode]))
                    serie = pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=np.float64)

                    if len(values) > 0:
                        if quality_flags:
                            q_flag = list(map(lambda x: x[3], seriesHolder[shortcode]))
                            serie = pd.Series(q_flag, index=pd.DatetimeIndex(timestamps), dtype=np.int32)
                            series.append(serie.to_frame('%s_%s' % (station, 'Q_FLAG')))
                            continue
                                # series.append(serie.to_frame('%s_%s_%s' % (station, 'quality_flag')))
                        sensors = list(set(list(map(lambda x: x[2], seriesHolder[shortcode]))))
                        serie = pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=np.float64)
                        if len(variables) == 1:
                            series.append(serie.to_frame('%s_%s' % (station, sensors[0])))
                        else:
                            series.append(serie.to_frame('%s_%s_%s' % (shortcode, station, sensors[0])))

                    # Clean up scope.
                    del values
                    del serie

                # Clean up memory.
                gc.collect()

            # Clean up.
            del seriesHolder
            gc.collect()

            # Merge all series together.
            if len(series) > 0:
                df = pd.concat(series, axis=1, sort=True)
                
            else:
                df = pd.DataFrame()
            
            

            # Clean up memory.
            del series
            gc.collect()
            # check if dataframe is empty
            if df.empty:
                # add the date range in the dataframe and the column as the station filled with NaN
                df = pd.DataFrame(index=pd.date_range(start=startDate, end=endDate, tz='UTC', freq=aggregate), columns=[f'{station}'])
                # remove the last row
                return df[:-1]
           
            else:
                # remove the last row 
                df = df[:-1] # lacks values for the last day
                return self.aggregate_variables(df, freq=aggregate)

    def multiple_measurements(self, 
                              stations_list, 
                              startDate, 
                              endDate, 
                              variables, 
                              dataset='controlled',
                              csv_file=None, 
                              aggregate='1D'):
        """
        Retrieves measurements for multiple stations within a specified date range.

        Parameters:
        -----------
        - stations_list (list): A list of strings containing the codes of the stations to retrieve data from.
        - startDate (str): The start date for the measurements, in the format 'yyyy-mm-dd'.
        - endDate (str): The end date for the measurements, in the format 'yyyy-mm-dd'.
        - variables (list): A list of strings containing the names of the variables to retrieve.
        - dataset (str): The name of the database to retrieve the data from. Default is 'controlled' alternatively 'raw' database.
        - csv_file (str, optional): pass the name of the csv file to save the data otherwise it will return the dataframe.
        - aggregate (bool): If True, aggregate the data per day; otherwise, return data in 5 minute interval.

        Returns:
        -----------
        - df (pandas.DataFrame): A DataFrame containing the aggregated data for all stations.

        Raises:
        -----------
        - ValueError: If stations_list is not a list.

        ### Example Usage:
        To retrieve precipitation data for stations in Kenya for the last week and save it as a csv file:
        ```python
        # Import the necessary modules
        from datetime import datetime, timedelta
        from filter_stations import retreive_data

        # An instance of the retreive_data class
        ret = retreive_data(apiKey, apiSecret, maps_key)

        # Get today's date
        today = datetime.now()

        # Calculate one week ago
        last_week = today - timedelta(days=7)

        # Format date as a string
        last_week_str = last_week.strftime('%Y-%m-%d')
        today_str = today.strftime('%Y-%m-%d')

        # Define the list of stations you want to retrieve data from example stations in Kenya
        stations = list(ret.get_stations_info(countrycode='KE')['code'])

        # Get the precipitation data for the stations in the list
        variables = ['pr']

        # retrieve the raw data for the stations, aggregate the data and save it as a csv file
        dataset = 'raw'
        aggregate = '1D'
        csv_file = 'Kenya_precipitation_data'

        # Call the multiple_measurements method to retrieve and aggregate data
        aggregated_data = ret.multiple_measurements(stations, last_week_str, 
                                                    today_str, variables, 
                                                    dataset, csv_file, aggregate)
        ```
        """
        if not isinstance(stations_list, list):
            raise ValueError('Pass in a list')

        error_dict = {}
        pool = mp.Pool(processes=mp.cpu_count())  # Use all available CPU cores

        try:
            results = []
            with tqdm(total=len(stations_list), desc='Retrieving data for stations') as pbar:
                for station in stations_list:
                    results.append(pool.apply_async(self.get_measurements, args=(station, startDate, endDate, variables, dataset, aggregate), callback=lambda _: pbar.update(1)))

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
        
    # multiple quality flags for multiple stations
    def multiple_qualityflags(self, stations_list, startDate, endDate, csv_file=None):
        """
        Retrieves and aggregates quality flag data for multiple stations within a specified date range.

        Parameters:
        -----------
        - stations_list (list): A list of station codes for which to retrieve data.
        - startDate (str): The start date in 'YYYY-MM-DD' format.
        - endDate (str): The end date in 'YYYY-MM-DD' format.
        - csv_file (str, optional): The name of the CSV file to save the aggregated data. Default is None.

        Returns:
        -----------
        - pandas.DataFrame or None: A DataFrame containing the aggregated quality flag data for the specified stations,
        or None if an error occurs.

        Raises:
            Exception: If an error occurs while retrieving data for a station.

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
    
    # get the anomalies data report
    def anomalies_report(self, start_date, end_date=None):
        """
        Retrieves anomaly reports for a specified date range.

        Parameters:
        -----------
        - start_date (str): The start date for the report in 'yyyy-mm-dd' format.
        - end_date (str, optional): The end date for the report in 'yyyy-mm-dd' format.
                                    If not provided, only data for the start_date is returned.

        Returns:
        -----------
        - pandas.DataFrame: A DataFrame containing anomaly reports with columns 'startDate',
                            'station_sensor', and 'level'. The 'startDate' column is used as the index.

        Raises:
        -----------
        - Exception: If there's an issue with the API request.

        Usage:
        -----------
        To retrieve anomaly reports for a specific date range:
        ```python
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        report_data = ret.anomalies_report(start_date, end_date)
        ```

        To retrieve anomaly reports for a specific date:
        ```
        start_date = '2023-01-01'
        report_data = ret.anomalies_report(start_date)
        ```
        """
        reqUrl = "https://datahub.tahmo.org/custom/sensordx/reports" # endpoint
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
            anomalies_data = pd.DataFrame(apiRequest.json()['qualityObjects'])
            level_2 = anomalies_data[(anomalies_data.level == 2) & (anomalies_data.type == 'sensordx')]
            level_2['station_sensor'] = level_2['stationCode'] + '_' + level_2['sensorCode']
            level_2 = level_2[['startDate', 'station_sensor', 'description', 'level']]
            level_2.startDate = pd.to_datetime([dateutil.parser.parse(i).strftime('%Y-%m-%d') for i in level_2['startDate']])
            level_2.set_index('startDate', inplace=True)
            level_2 = level_2.sort_index()
            # print(level_2)
            try:
                if end_date:
                    return level_2.loc[start_date:end_date]
                else:
                    return level_2.loc[start_date]
            except KeyError as e:
                return e
        else:
            return self.__handleApiError(apiRequest)
        
    # get the ground truth data
    def ground_truth(self, start_date, end_date=None):
        """
        Retrieves ground truth data for a specified date range.

        Parameters:
        -----------
        - start_date (str): The start date for the report in 'yyyy-mm-dd' format.
        - end_date (str, optional): The end date for the report in 'yyyy-mm-dd' format.
                                    If not provided, only data for the start_date is returned.

        Returns:
        -----------
        - pandas.DataFrame: A DataFrame containing ground truth data with columns 'startDate',
                            'station_sensor',  'description' and 'level'. The 'startDate' column is used as the index.

        Raises:
        -----------
        - Exception: If there's an issue with the API request.

        Usage:
        -----------
        To retrieve ground truth data for a specific date range:
        ```python
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        report_data = ret.ground_truth(start_date, end_date)
        ```

        To retrieve ground truth data for a specific date:
        ```
        start_date = '2023-01-01'
        report_data = ret.ground_truth(start_date)
        ```
        """
        reqUrl = "https://datahub.tahmo.org/custom/sensordx/reports" # endpoint
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
            reports = pd.DataFrame(apiRequest.json()['qualityObjects'])
            reports = reports[reports.level != 2][['startDate', 'endDate', 'stationCode', 'sensorCode', 'description', 'level']]
            reports['station_sensor'] = reports.stationCode + '_' + reports.sensorCode
            reports = reports.drop(['stationCode', 'sensorCode'], axis=1)

            # convert the start and end date to datetime format
            reports['startDate'] = pd.to_datetime(reports['startDate']).dt.tz_localize(None)
            reports['endDate'] = pd.to_datetime(reports['endDate']).dt.tz_localize(None)
            # convert start_date string to datetime format
            start_date_dt = pd.to_datetime(start_date).tz_localize(None)

            try:
                if end_date is None:
                    # check for the date
                    def check_date(row):
                        if row.startDate <= start_date_dt and row.endDate >= start_date_dt:
                            return start_date
                    reports['Date'] = reports.apply(check_date, axis=1)
                    reports = reports.dropna()
                    reports = reports[['Date', 'station_sensor', 'description', 'level']]
                    reports.set_index('Date', inplace=True)
                    return reports
                else:
                    # convert end_date string to datetime format
                    end_date_dt = pd.to_datetime(end_date).tz_localize(None)
                    
                    # Define a function to check if a date is within a range
                    def check_date(row, date):
                        return row['startDate'] <= date and row['endDate'] >= date
                    reports_list = []
                    # Iterate over the date range
                    for single_date in pd.date_range(start_date, end_date):
                        # Filter the reports for the current date
                        filtered_reports = reports[reports.apply(check_date, axis=1, date=single_date)]
                        
                        # Add the current date as a new column
                        filtered_reports['Date'] = single_date
                        
                        # Append the filtered reports to the list
                        reports_list.append(filtered_reports)
                    filtered_reports_df = pd.concat(reports_list)

                    # Drop the startDate and endDate columns
                    filtered_reports_df = filtered_reports_df.drop(['startDate', 'endDate'], axis=1)
                    # Set the index to the Date column
                    filtered_reports_df.set_index('Date', inplace=True)
                    return filtered_reports_df

            except KeyError as e:
                return e
                    


        else:
            return self.__handleApiError(apiRequest)

'''
A specific class to evaluate and validate the water level data using TAHMO Stations
To be used as it is to maintain flow
'''
class pipeline(retreive_data):
    # inherit from retrieve_data class
    def __init__(self, apiKey, apiSecret, api_key):
        super().__init__(apiKey, apiSecret, api_key)

    
    # given the radius and the longitude and latitude of the gauging station, return the stations within
    def stations_within_radius(self, radius, latitude, longitude, df=False):
        """
    Retrieves stations within a specified radius from a given latitude and longitude.

    Parameters:
    -----------
    - radius (float): Radius (in kilometers) within which to search for stations.
    - latitude (float): Latitude of the center point.
    - longitude (float): Longitude of the center point.
    - df (bool, optional): Flag indicating whether to return the result as a DataFrame. Defaults to False.

    Returns:
    - DataFrame or list: DataFrame or list containing the stations within the specified radius. If df is True, 
    a DataFrame is returned with the columns 'code', 'location.latitude', 'location.longitude', and 'distance'. 
    If df is False, a list of station codes is returned.

    """
        stations  = super().get_stations_info()
        stations['distance'] = stations.apply(lambda row: hs.haversine((latitude, longitude), (row['location.latitude'], row['location.longitude'])), axis=1)
        infostations = stations[['code', 'location.latitude','location.longitude', 'distance']].sort_values('distance')
        if df:
            return infostations[infostations['distance'] <= radius]
        else:
            return infostations[infostations['distance'] <= radius].code.values
        
        
    def stations_data_check(self, stations_list, percentage=1, start_date=None, end_date=None, data=None, variables=['pr'], csv_file=None):
        """
        Performs a data check on the stations' data and returns the stations with a percentage of missing data below a threshold.

        Parameters:
        -----------
        - stations_list (list): List of station names or IDs.
        - percentage (float, optional): Threshold percentage of missing data. Defaults to 1 (i.e., 0% missing data allowed).
        - start_date (str, optional): Start date for the data range in the format 'YYYY-MM-DD'. Defaults to None.
        - end_date (str, optional): End date for the data range in the format 'YYYY-MM-DD'. Defaults to None.
        - data (DataFrame, optional): Preloaded data for the stations. Defaults to None.
        - variables (list, optional): List of variables to consider for the data check. Defaults to ['pr'].
        - csv_file (str, optional): File name for saving the data as a CSV file. Defaults to None.

        Returns:
        -----------
        - DataFrame: DataFrame containing the stations' data with less than the specified percentage of missing data.

        """
        if data is None:
            data = super().multiple_measurements(stations_list, startDate=start_date, endDate=end_date, variables=variables, csv_file=csv_file)

        # Check the percentage of missing data and return the stations with less than the percentage of missing data
        data.index = data.index.astype('datetime64[ns]')
        data = data.dropna(axis=1, thresh=int(len(data) * percentage))
        data.to_csv(f'{csv_file}.csv')
        return data
    
    def calculate_lag(self, weather_stations_data, water_level_data, lag=3, above=None, below=None):
        """
        Calculates the lag and coefficient of correlation between weather station data
        and water level data, identifying stations with positive correlations.

        Parameters:
        ------------
        - weather_stations_data (DataFrame): A DataFrame containing weather
        station data columns for analysis.
        - water_level_data (Series): A time series of water level data used for
        correlation analysis.
        - lag (int): The maximum lag, in hours, to consider for correlation.
        Default is 3 hours.
        - above (float or None): If specified, stations with correlations and lags
        above this threshold are identified.
        - below (float or None): If specified, stations with correlations and lags
        below this threshold are identified.

        Returns:
        ------------
        - above_threshold_lag (dict): A dictionary where keys represent weather station
        column names, and values represent the lag in hours if positive correlation
        exceeds the specified threshold (above).
        - below_threshold_lag (dict): A dictionary where keys represent weather station
        column names, and values represent the lag in hours if positive correlation
        falls below the specified threshold (below).
        """
        above_threshold_lag = dict()
        below_threshold_lag = dict()
        for cols in weather_stations_data.columns:
            # check for positive correlation if not skip the column
            if weather_stations_data[cols].corr(water_level_data['water_level']) <= 0:
                continue
            # get the lag and the coefficient for columns with a positive correlation
            coefficient_list = list(sm.tsa.stattools.ccf(weather_stations_data[cols], water_level_data['water_level']))    
            a = np.argmax(coefficient_list)
            b = coefficient_list[a] 
            # print(f'{cols} has a lag of {a}')
            # print(f'{cols} has a coefficient of {b}')
            # print('-----------------------')
            if a > lag:
                above_threshold_lag[cols] = a
            elif a <= lag:
                below_threshold_lag[cols] = a
        if above:
            return above_threshold_lag
        elif below:
            return below_threshold_lag
        else:
            return above_threshold_lag, below_threshold_lag
        
    def shed_stations(self, weather_stations_data, water_level_data,
                        gauging_station_coords, radius, lag=3,
                        percentage=1):
        """
        Filters and processes weather station data to identify stations
        potentially contributing to water level changes above or below
        specified thresholds.

        Parameters:
        ------------
        - weather_stations_data (DataFrame): A DataFrame containing weather
        station data over a specific date range.
        - water_level_data (Series): A time series of water level data
        corresponding to the same date range as weather_station_data.
        - gauging_station_coords (tuple): A tuple containing latitude and
        longitude coordinates of the gauging station.
        - radius (float): The radius in kilometers for identifying nearby
        weather stations.
        - lag (int): The time lag, in hours, used for correlation analysis.
        Default is 3 hours.
        - percentage (float): The minimum percentage of valid data required
        for a weather station to be considered. Default is 1 (100%).
        - above (float or None): The threshold above which water level changes
        are considered significant. If provided, stations contributing to
        changes above this threshold are identified.
        - below (float or None): The threshold below which water level changes
        are considered significant. If provided, stations contributing to
        changes below this threshold are identified.

        Returns:
        ------------
        - above_threshold_lag (list): List of weather stations with
            positive correlations and lagged changes above the specified threshold.
        - below_threshold_lag (list): List of weather stations with
            positive correlations and lagged changes below the specified threshold.
        
        Usage:
        ------------
        Get the TAHMO stations that correlate with the water level data
        ```python
        import pandas as pd
        from filter_stations import pipeline

        # An instance of the pipeline class
        pipe = pipeline(apiKey, apiSecret, maps_key)

        # load the water level data and the weather stations data
        water_level_data = pd.read_csv('water_level_data.csv')
        weather_stations_data = pd.read_csv('weather_stations_data.csv') 

        # get the coordinates of the gauging station
        gauging_station_coords = (-0.416, 36.951)

        # get the stations within a radius of 200km from the gauging station
        radius = 200
        
        # get the stations that correlate with the water level data
        above_threshold_lag, below_threshold_lag = pipe.shed_stations(weather_stations_data, water_level_data, 
                                                                      gauging_station_coords, radius, 
                                                                      lag=3, percentage=1)
        ```
        
        """
        # Filter the date range based on the water level data from first day of the water level data to the last day of the water level data
        weather_stations_data = weather_stations_data.loc[water_level_data.index[0]:water_level_data.index[-1]]
        # Filter the weather stations based on the radius
        lat, lon = gauging_station_coords[0], gauging_station_coords[1]
        weather_stations_data_list = self.stations_within_radius(radius, lat, lon, df=False)
        # get stations without missing data or the percentage of stations with missing data
        weather_stations_data_filtered = self.stations_data_check(stations_list=weather_stations_data_list,
                                                                percentage=percentage,
                                                                data=weather_stations_data)
        # Check the sum of each column and drop columns with a sum of zero this is if the sum of water level is not equal to zero
        weather_stations_data_filtered = weather_stations_data_filtered.loc[:, weather_stations_data_filtered.sum() != 0]

        # Filter the weather stations based on the lag and positive correlation
        above_threshold_lag, below_threshold_lag = self.calculate_lag(weather_stations_data_filtered, 
                                                                      water_level_data, lag=lag)

        return above_threshold_lag, below_threshold_lag
            
        
    def plot_figs(self, weather_stations, water_list, threshold_list, save=False, dpi=500, date='11-02-2021'):
        """
        Plots figures showing the relationship between rainfall and water level/stage against time.

        Parameters:
        -----------
        - weather_stations (DataFrame): DataFrame containing weather station data.
        - water_list (list): List of water levels/stages.
        - threshold_list (list): List of columns in the weather_stations DataFrame to plot.
        - save (bool, optional): Flag indicating whether to save the figures as PNG files. Defaults to False.
        - dpi (int, optional): Dots per inch for saving the figures. Defaults to 500.
        - date (str, optional): Start date for plotting in the format 'dd-mm-yyyy'. Defaults to '11-02-2021'.

        Returns:
        -----------
        - Displays the images of the plots. and if save is set to true saves the images in the current directory.

        
        <div align="center">
          <img src="water_level_pipeline_15_1.png" alt="Muringato" width="80%">
        </div>

        """
        start_date = datetime.datetime.strptime(date, "%d-%m-%Y")
        end_date = start_date + datetime.timedelta(len(water_list)-1)
        # weather_stations = weather_stations.set_index('Date')
        df_plot = weather_stations[start_date:end_date]
        df_plot = df_plot[threshold_list].reset_index()
        df_plot.rename(columns={'index':'Date'}, inplace=True)
        
        
        plt.rcParams['figure.figsize'] = (15, 9)
        print('Begin plotting!')
        
        for cols in df_plot.columns[1:]:
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel(f'Time', fontsize=24, weight='bold')
            ax1.set_ylabel(f'Rainfall {cols} (mm)', color=color, fontsize=24, weight='bold')
            ax1.bar(pd.to_datetime(df_plot['Date'], format="%d/%m/%Y"), df_plot[f'{cols}'], color=color, width=4, alpha=1.0)
            ax1.tick_params(axis='y', labelcolor=color, labelsize=24)
            ax1.tick_params(axis='x')
            ax1.set_xticklabels(df_plot['Date'], fontsize=21, weight='bold')
            ax1.grid(color='gray', linestyle='--', linewidth=0.8)
            ax1.set(facecolor="white")
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:red'
            ax2.set_ylabel('Water level/Stage (m)', color=color, fontsize=24, weight='bold')
            ax2.plot(pd.to_datetime(df_plot['Date'], format="%d/%m/%Y"), water_list, color=color, linewidth=4)
            ax2.tick_params(axis='y', labelcolor=color, labelsize=24)
            ax2.set(facecolor="white")
            plt.title('Stage and Rainfall against Time', fontsize=22, weight='bold')

            date_form = DateFormatter("%m-%y")
            ax1.xaxis.set_major_formatter(date_form)
            fig.tight_layout()

            if save:
                fig.savefig(f'{cols}.png', dpi=dpi)


# Move the functions to a class
class Filter(pipeline):
    # inherit from retrieve_data class
    def __init__(self, apiKey, apiSecret, api_key):
        super().__init__(apiKey, apiSecret, api_key)
    
    def get_stations_info(self, station=None, multipleStations=[], countrycode=None):
        return super().get_stations_info(station, multipleStations, countrycode)
        
    # Get the centre point of the address
    def centre_point(self, address):
        """
        This method retrieves the latitude and longitude coordinates of a given address using the Google Maps Geocoding API.
        
        Parameters:
        -----------
        - address : str
            The address of the location you want to retrieve the coordinates for.
        - api_key : str
            Your Google Maps Geocoding API key.
                
        Returns:
        --------
        - Tuple (float, float) or None
            The latitude and longitude coordinates of the location if found, or None if the address is not found.
        """
        base_url = 'https://maps.googleapis.com/maps/api/geocode/json'
        params = {
            'address': address,
            'key': self.api_key,
        }
        
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code == 200 and data.get('results'):
            # If the request was successful and results were found
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            # If the address is not found or there was an error
            print(f"Error while accessing Google Maps Geocoding API: {data.get('error_message', 'Unknown Error')}")
            return None
        
    # Get the new radius of the address
    def calculate_new_point(self, lat, lon, distance, bearing):
        """
        Calculates a new geographic point based on the given latitude, longitude,
        distance and bearing.

        Parameters:
        -----------
        - lat (float): The latitude of the starting point in decimal degrees.
        - lon (float): The longitude of the starting point in decimal degrees.
        - distance (float): The distance in kilometers from the starting point to the new point.
        - bearing (float): The bearing in degrees from the starting point to the new point,
            measured clockwise from true north.

        Returns:
        -----------
        - Tuple[float, float]: A tuple containing the latitude and longitude of the new point,
        respectively, in decimal degrees.
        """
        distance = distance * 1000
        # Convert degrees to radians
        lat = math.radians(lat)
        lon = math.radians(lon)
        bearing = math.radians(bearing)
        
        # Earth radius in meters
        R = 6371000
        
        # Calculate new latitude
        lat2 = math.asin(math.sin(lat) * math.cos(distance / R) +
                        math.cos(lat) * math.sin(distance / R) * math.cos(bearing))
                        
        # Calculate new longitude
        lon2 = lon + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat),
                                math.cos(distance / R) - math.sin(lat) * math.sin(lat2))
        
        # Convert back to degrees
        lat2 = math.degrees(lat2)
        lon2 = math.degrees(lon2)
        
        return lat2, lon2
    

    # Get the bounding box of the address
    def compute_filter(self, lat, lon, distance):
        """
        Calculates the bounding box coordinates for a given location and distance.

        Parameters:
        -----------
        - lat (float): The latitude of the location.
        - lon (float): The longitude of the location.
        - distance (float): The distance from the location, in kilometers, to the edge of the bounding box.

        Returns:
        -----------
        - A tuple containing four floats representing the bounding box coordinates: (min_lat, min_lon, max_lat, max_lon).
        """
        points = []
        g1 = []
        for i in range(0, 360, 45):
            points.append(self.calculate_new_point(lat, lon, distance, i))
        g1 = [min(p[0] for p in points), min(p[1] for p in points)]
        g2 = [max(p[0] for p in points), max(p[1] for p in points)]
        # print(g1, '\n', g2)
        return g1[0], g1[1], g2[0], g2[1]

    # Get the minimum and maximum latitude and longitude of the address


    def filter_stations(self, address, distance, startDate=None, endDate=None, csvfile='pr_clog_flags.csv'):
        """
        This method filters weather station data within a certain distance from a given address.
        
        Parameters:
        -----------
        - address (str): Address to center the bounding box around.
        - distance (float): The distance (in kilometers) from the center to the edge of the bounding box.
        - startDate (str): The start date for filtering the weather station data in the format 'YYYY-MM-DD'.
        - endDate (str): The end date for filtering the weather station data in the format 'YYYY-MM-DD'.
        - csvfile (str): The name of the csv file containing the weather station data.
        
        Returns:
        -----------
        - pandas.DataFrame: The filtered weather station data within the bounding box.
        """   
        centre = self.centre_point(address)
        lat, lon = float(centre[0]), float(centre[1])  
        stations = super().stations_within_radius(distance, lat, lon, df=False)
        # lat, lon = self.centre_point(address)
        # min_lat, min_lon, max_lat, max_lon = self.compute_filter(float(lat), float(lon), distance)
        # stations = super().get_stations_info()
        # bounds = list(stations['code'][(stations['location.longitude'] >= min_lon)
        #                                 & (stations['location.longitude'] <= max_lon)
        #                                 & (stations['location.latitude'] >= min_lat)
        #                                     & (stations['location.latitude'] <= max_lat)])
        
        # read the csv file
        ke_chec = pd.read_csv(csvfile)
        ke_chec.Date = ke_chec.Date.astype('datetime64[ns]')
        # print(ke_chec.info())

        # ke_chec = ke_chec.set_index('Date')
        if startDate and endDate:
            startdate = dateutil.parser.parse(startDate)
            enddate = dateutil.parser.parse(endDate)
            begin = ke_chec['Date'][ke_chec['Date'] == startdate].index.to_numpy()[0]
            end = ke_chec['Date'][ke_chec['Date'] == enddate].index.to_numpy()[0]
            ke_chec = ke_chec.iloc[begin:end+1]
            ke_chec = ke_chec.set_index('Date')

            return ke_chec[[i for i in ke_chec.columns if i.split('_')[0] in stations]]
        else:
            ke_chec = ke_chec.set_index('Date')
            return ke_chec[[i for i in ke_chec.columns if i.split('_')[0] in stations]]


    # A list of filtered stations
    def filter_stations_list(self, address, distance=100):
        """
        Filters stations based on their proximity to a given address and returns a list of station codes that fall within the specified distance.
        
        Parameters:
        -----------
        - address (str): Address to filter stations by.
        - distance (float, optional): Maximum distance (in kilometers) between the stations and the address. Default is 100 km.
        
        Returns:
        -----------
        - List of station codes that fall within the specified distance from the given address.
        """
        return list(set([i.split('_')[0] for i in self.filter_stations(f'{address}', distance).columns if i.split('_')[-1] != 'clogFlag']))
    
    def stations_region(self, region, plot=False):
        """
        Subsets weather stations by a specific geographical region and optionally plots them on a map with a scale bar.

        Parameters:
        -----------
        - region (str): The name of the region to subset stations from (47 Kenyan counties).
        - plot (bool, optional): If True, a map with stations and a scale bar is plotted. Default is False.

        Returns:
        -----------
        - list or None: If plot is False, returns a list of station codes in the specified region. Otherwise, returns None.

        Usage:
        -----------
        To get a list of station codes in the 'Nairobi' region without plotting:
        ```
        fs = Filter(api_key, api_secret, maps_key)  # Create an instance of your class
        station_list = fs.stations_region('Nairobi')
        ```

        To subset stations in the 'Nairobi' region and display them on a map with a scale bar:
        ```
        fs = Filter(api_key, api_secret, maps_key)  # Create an instance of your class
        fs.stations_region('Nairobi', plot=True)
        ```
        <div align="center">
          <img src="nairobi_region.png" alt="Nairobi Region Map" width="80%">
        </div>
        """
        # Handle different ways of writing region
        region = region.title()
        
        # get the path of the files
        adm0_path = os.path.join(module_dir, 'geo', 'gadm41_KEN_1.shp')
        adm3_path = os.path.join(module_dir, 'geo', 'gadm41_KEN_3.shp')
        # read the map data from the shapefile the greater and smaller region
        gdf_adm0 = gpd.read_file(adm0_path)
        gdf_adm3 = gpd.read_file(adm3_path)
        # get the stations metadata
        stations = super().get_stations_info()[['code', 'location.latitude', 'location.longitude']]

        # subset by the particular region
        stations['test'] = stations.apply(lambda row: gdf_adm0[gdf_adm0.NAME_1 == f'{region}']
                                        [['geometry']].contains(Point(row['location.longitude'], 
                                                                        row['location.latitude'])), axis=1)
        stations = stations[stations.test]
        scale_factor = 432.16

        # return the list of stations if plotting is false
        if plot:
            station_geometry = [Point(xy) for xy in zip(stations['location.longitude'], 
                                                stations['location.latitude'])]

            # Create a GeoDataFrame from the station data
            station_gdf = gpd.GeoDataFrame(stations, 
                                        geometry=station_geometry, 
                                        crs=gdf_adm0.crs)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf_adm3[gdf_adm3.NAME_1 == f'{region}'].plot(ax=ax, color='gray')
            # if station_grid has no value
            try:
                station_gdf.plot(ax=ax, color='blue', marker='o', markersize=20)  # Adjust marker size as needed
            except ValueError:
                pass

            # Add the scale bar using mplscale
            ax.add_artist(ScaleBar(scale_factor, units='km', 
                                location='lower right', 
                                length_fraction=0.35))

            plt.title(f'{region} region')
            plt.show()
        else:
            return list(stations['code'])
    
    # remove columns with all zeros
    def remove_zero_columns(self, df):
        """
        Removes columns with all zeros from a DataFrame.

        Parameters:
        -----------
        - df (DataFrame): The DataFrame to remove columns from.

        Returns:
        -----------
        - DataFrame: The DataFrame with columns containing all zeros removed.
        """
        return df.loc[:, (df != 0).any(axis=0)]
    
    # filter by precipitation data from BigQuery
    def filter_pr(self, start_date, end_date, country=None, region=None,
                    radius=None, multiple_stations=None, station=None):
        """
        Retrieves precipitation data from BigQuery based on specified parameters.

        Parameters:
        -----------
        - start_date (str): Start date for data query.
        - end_date (str): End date for data query.
        - country (str): Country name for filtering stations.
        - region (str): Region name for filtering stations.
        - radius (str): Radius for stations within a specified region.
        - multiple_stations (str): Comma-separated list of station IDs.
        - station (str): Single station ID for data filtering.

        Returns:
        -----------
        - pd.DataFrame: A Pandas DataFrame containing the filtered precipitation data.
        
        Usage:
        -----------
        To get precipitation data for a specific date range:
        ```python
        fs = Filter(api_key, api_secret, maps_key)  # Create an instance of your class
        start_date = '2021-01-01'
        end_date = '2021-01-31'
        pr_data = fs.filter_pr(start_date, end_date)
        ```
        To get precipitation data for a specific date range and country:
        ```python
        fs = Filter(api_key, api_secret, maps_key)  # Create an instance of your class
        start_date = '2021-01-01'
        end_date = '2021-01-31'
        country = 'Kenya'
        pr_data = fs.filter_pr(start_date, end_date, country=country)
        ```
        To get precipitation data for a specific date range and region:
        ```python
        fs = Filter(api_key, api_secret, maps_key)  # Create an instance of your class
        start_date = '2021-01-01'
        end_date = '2021-01-31'
        region = 'Nairobi'
        pr_data = fs.filter_pr(start_date, end_date, region=region)
        ```
        To get precipitation data for a specific date range and region with a radius:
        ```python
        fs = Filter(api_key, api_secret, maps_key)  # Create an instance of your class
        start_date = '2021-01-01'
        end_date = '2021-01-31'
        region = 'Nairobi'
        radius = 100
        pr_data = fs.filter_pr(start_date, end_date, region=region, radius=radius)
        ```
        To get precipitation data for a specific date range and multiple stations:
        ```python
        fs = Filter(api_key, api_secret, maps_key)  # Create an instance of your class
        start_date = '2021-01-01'
        end_date = '2021-01-31'
        multiple_stations = ['TA00001', 'TA00002', 'TA00003']
        pr_data = fs.filter_pr(start_date, end_date, multiple_stations=multiple_stations)
        ```
        To get precipitation data for a specific date range and a single station:
        ```python
        fs = Filter(api_key, api_secret, maps_key)  # Create an instance of your class
        start_date = '2021-01-01'
        end_date = '2021-01-31'
        station = 'TA00001'
        pr_data = fs.filter_pr(start_date, end_date, station=station)
        ```

        """
        print('Retrieving precipitation data from BigQuery...')
        # print(self.apiKey, self.apiSecret)
        base_url = f"https://us-central1-tahmo-quality-control.cloudfunctions.net/retrieve-from-bigquery?api_key={self.apiKey}&api_secret={quote(self.apiSecret)}&start_date={start_date}&end_date={end_date}"
        if country:
            base_url += f'&country={country}'
        elif region and radius is None:
            base_url += f'&region={region}'
        elif region and radius:
            base_url += f'&region={region}&radius={radius}'
        elif multiple_stations:
            base_url += f'&multiple_stations={multiple_stations}'
        elif station:
            base_url += f'&station={station}'
        # print(base_url)
        headersList = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client (https://www.thunderclient.com)" 
        }

        payload = ""
        apiRequest = requests.get(f'{base_url}',
                                    params={},
                                    auth=requests.auth.HTTPBasicAuth(
                                    self.apiKey,
                                    self.apiSecret
                                )
        )

        
        if apiRequest.status_code == 200:
            response =  apiRequest.json()
            # print(apiRequest.text)
            return pd.read_json(response['data'])
        else:
            return self._retreive_data__handleApiError(apiRequest) 

    # get clogs for a certain duration based on quality objects file
    def clogs(self, startdate, enddate, flags_json='qualityobjects.json', as_csv=False, csv_file=None):
        """
        Generate clog flags DataFrame based on start and end dates.

        Parameters:
        -----------
        - startdate (str): Start date in 'YYYY-MM-DD' format.
        - enddate (str): End date in 'YYYY-MM-DD' format.
        - flags_json (str, optional): Path to the JSON file containing clog flags data. Defaults to 'qualityobjects.json'.
        - questionable (bool, optional): Whether to return questionable clog flags. Defaults to False.
        - as_csv (bool, optional): Whether to save the resulting DataFrame as a CSV file. Defaults to False.
        - csv_file (str, optional): Name of the CSV file to save. Only applicable if as_csv is True. Defaults to None.

        Returns:
        -----------
        - pandas.DataFrame: DataFrame containing the clog flags.

        """

        json_data = pd.read_json(flags_json)
        startdate = datetime.datetime.strptime(startdate, '%Y-%m-%d')
        enddate = datetime.datetime.strptime(enddate, '%Y-%m-%d')

        # merge the sensorcode and the stationcode to equate to the station_sensor format 
        json_data['station_sensor'] = json_data['stationCode'] + '_' + json_data['sensorCode']
        # convert the start and end time to datetime format datetime[ns]
        json_data['startDate'] = json_data['startDate'].astype('datetime64[ns]')
        json_data['endDate'] = pd.to_datetime([dateutil.parser.parse(i).strftime('%Y-%m-%d') for i in json_data['endDate']])

        json_data = json_data.drop(['stationCode', 'sensorCode'], axis=1)
        json_data = json_data[['description', 'startDate', 'endDate', 'station_sensor']]

        other_failure = list(json_data[json_data['description'].str.contains('batter')].index)
        clog = list(json_data[~json_data['description'].str.contains('batter')].index)

        clog_flags = pd.DataFrame(pd.date_range(startdate, enddate, freq='D'), columns=['Date'])

        clogs_dict = dict()
        for i in json_data.station_sensor.unique():
            clogs_dict[f'{i}_clogFlags'] = [np.nan for i in range(len(clog_flags))]

        flags_df = pd.concat([clog_flags, pd.DataFrame(clogs_dict)], axis=1)

        def subset_flags_df(startdate, enddate, column, val):
            """
            Update a subset of flags in the DataFrame with a specific value.

            Args:
                startdate (datetime): Start date of the subset.
                enddate (datetime): End date of the subset.
                column (str): Name of the column to update.
                val: Value to fill in the specified column.

            Returns:
                None

            """
            # print(f'Updating {column} from {startdate} to {enddate} to {val}')
            flags_df.loc[((flags_df['Date'] >= startdate) & (flags_df['Date'] <= enddate)), column] = val

        for ind, row in json_data.iterrows():
            if ind in other_failure:
                if row['startDate'] >= startdate and row['endDate'] <= enddate:
                    subset_flags_df(row['startDate'], row['endDate'], f"{row['station_sensor']}_clogFlags", 2)
                elif row['startDate'] >= startdate and row['endDate'] > enddate:
                    subset_flags_df(row['startDate'], enddate, f"{row['station_sensor']}_clogFlags", 2)
                elif row['startDate'] < startdate and row['endDate'] <= enddate:
                    subset_flags_df(startdate, row['endDate'], f"{row['station_sensor']}_clogFlags", 2)
                elif row['startDate'] < startdate and row['endDate'] > enddate:
                    subset_flags_df(startdate, enddate, f"{row['station_sensor']}_clogFlags", 2)
            elif ind in clog:
                if row['startDate'] >= startdate and row['endDate'] <= enddate:
                    subset_flags_df(row['startDate'], row['endDate'], f"{row['station_sensor']}_clogFlags", 1)
                elif row['startDate'] >= startdate and row['endDate'] > enddate:
                    subset_flags_df(row['startDate'], enddate, f"{row['station_sensor']}_clogFlags", 1)
                elif row['startDate'] < startdate and row['endDate'] <= enddate:
                    subset_flags_df(startdate, row['endDate'], f"{row['station_sensor']}_clogFlags", 1)
                elif row['startDate'] < startdate and row['endDate'] > enddate:
                    subset_flags_df(startdate, enddate, f"{row['station_sensor']}_clogFlags", 1)

        flags_df = flags_df.set_index('Date')
        flags_df = flags_df.reindex(sorted(flags_df.columns), axis=1)

        if as_csv:
            if csv_file is not None:
                flags_df.to_csv(f'{csv_file}.csv', index=True)
                return flags_df
            else:
                flags_df.to_csv('clog_flags.csv', index=True)
                return flags_df
        else:
            return flags_df

# A different class for visualisations
class Interactive_maps(retreive_data):
    # inherit from retrieve_data class
    def __init__(self, apiKey, apiSecret, api_key):
        super().__init__(apiKey, apiSecret, api_key)

    def draw_map(self, map_center):
        """
        Creates a Folium map centered on the specified location and adds markers for each weather station in the area.

        Parameters:
        -----------
        - map_center: a tuple with the latitude and longitude of the center of the map

        Returns:
        -----------
        - A Folium map object
        """

        #  retrieve the stations data
        stations = super().get_stations_info()[['code', 'location.longitude', 'location.latitude']]
        # create a map centered on a specific location

        my_map = folium.Map(location=map_center, min_zoom=5, max_zoom=30, zoom_start=12, tiles='cartodbpositron')
        for _, row in stations.iterrows():
            folium.Marker([row['location.latitude'], row['location.longitude']], tooltip=row['code']).add_to(my_map)


        # fit the map bounds to the markers' locations
        marker_locations = [[row['location.latitude'], row['location.longitude']] for _, row in stations.iterrows()]
        my_map.fit_bounds(marker_locations)

        # display the map
        return my_map


    def animation_image(self, sensors,  start_date, end_date, day=100, T=10, interval=500, data=None):
        '''
        Creates an animation of pollutant levels for a given range of days and valid sensors.

        Parameters:
        -----------
        - data (DataFrame): A pandas DataFrame containing station data defaults to none reads pr_clog_flags if none.
        - sensors (list): A list of valid sensor names.
        - day (int): The starting day of the animation (default is 100).
        - T (int): The range of days for the animation (default is 10).
        - interval (int): The interval between frames in milliseconds (default is 500).

        Returns:
        -----------
        - HTML: An HTML object containing the animation.
        '''
        if not data:
            data = pd.read_csv('pr_clog_flags.csv')
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

        
        data1 = np.array(data[sensors].iloc[day:day+1]).T
        data1 /= np.max(data1)
        # Create figure and axis
        fig, ax = plt.subplots()

        # Define function to update matrix data
        def update(i):
            # Generate new data
            new_data =  np.array(data[sensors].iloc[int(day):int(day+1+i)]).T
            new_data /= np.max(new_data)

            # Update matrix data
            im.set_array(new_data)

            return [im]

        # Create matrix plot
        im = ax.imshow(data1, aspect='auto', interpolation=None)

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=range(T), interval=interval, blit=False)

        plt.close()

        return HTML(ani.to_jshtml())
    
    # create animation grid
    def animation_grid(self, mu_pred, xi, xj, valid_station_df, clogged_station_df, T=10):
        """
        Creates an animation of the predicted data on a grid over time.

        Parameters:
        -----------
        - mu_pred (ndarray): The predicted data on a grid over time.
        - xi (ndarray): The x-coordinates of the grid.
        - xj (ndarray): The y-coordinates of the grid.
        - valid_station_df (DataFrame): A DataFrame containing the information of the valid stations.
        - clogged_station_df (DataFrame): A DataFrame containing the information of the clogged stations.
        - T (int): The number of time steps.

        Returns:
        -----------
        - HTML: The animation as an HTML object.
        
        The animation as an MP4 file

        <video width="320" height="240" controls>
            <source src="animation.mp4" type="video/mp4">
        </video>

        """
        fig, ax = plt.subplots()

        def animate(t):
            ax.clear()
            ax.set_title('Time step {}'.format(t))
            ax.pcolor(xi, xj, mu_pred[:, t:t+1].reshape(xi.shape), shading='auto')
            ax.plot(valid_station_df['Longitude'], valid_station_df['Latitude'], '.')
            ax.plot(clogged_station_df['Longitude'], clogged_station_df['Latitude'], 'r.')

        ani = animation.FuncAnimation(fig, animate, frames=T, interval=500, blit=False)
        plt.close()
        
        return HTML(ani.to_jshtml())



    def plot_station(self, ws, df_rainfall):
        """
        Plot the rainfall data for a specific weather station.

        Parameters:
        -----------
        - ws: string, the code of the weather station to plot
        - df_rainfall: DataFrame, a pandas DataFrame with rainfall data

        Returns:
        -----------
        - None if no data is available for the specified station
        - a Matplotlib figure showing rainfall data for the specified station otherwise
        """
        # filter columns based on station code and type (sensor or clog flag)
        sensors = [x for x in list(df_rainfall.keys()) if ws in x and 'clog' not in x]
        clog_flags = [x for x in list(df_rainfall.keys()) if ws in x and 'clog' in x]

        # check if any data is present
        if not sensors and not clog_flags:
            return None

        # create subplots
        try:
            fig, ax = plt.subplots(nrows=len(sensors)+len(clog_flags), ncols=1, figsize=(13,13))
        except:
            return None

        # plot data for sensors and clog flags
        
        # plot data for sensors and clog flags
        df_rainfall[sensors].plot(subplots=True, ax=ax[:len(sensors)])
        df_rainfall[clog_flags].plot(subplots=True, ax=ax[len(sensors):])

        # adjust spacing between subplots
        plt.subplots_adjust(hspace=0.5)

        # add x and y axis labels and title
        plt.xlabel('Time')
        plt.ylabel('Rainfall')
        plt.suptitle('Rainfall data for sensors and clog flags at station {}'.format(ws))
        plt.close()
        return fig


    # Encode the image
    def encode_image(self, ws, df_rainfall):
        """
        Encodes a station's rainfall data plot as a base64-encoded image.

        Parameters:
        -----------
        - ws (str): the code for the station to encode the image for
        - df_rainfall (pandas.DataFrame): a DataFrame containing the rainfall data for all stations
        
        Returns:
        -----------
        - str: a string containing an HTML image tag with the encoded image data, or a message indicating no data is available for the given station
        """
        figure = self.plot_station(ws, df_rainfall)
        if figure is not None:
            figure.tight_layout()
            buf = BytesIO()
            figure.savefig(buf, format='png')
            plt.close() # close the figure object to remove the subplot
            buf.seek(0)
            # Encode the image data as base64
            image_data = base64.b64encode(buf.read()).decode('utf-8')
            return '<img src="data:image/png;base64,{}">'.format(image_data)
        else:
            return 'No data available for station {}'.format(ws)



    def get_map(self, subset_list, start_date=None, end_date=None, data_values=False, csv_file='pr_clog_flags.csv', min_zoom=8, max_zoom=11, width=2000, height=2000, png_resolution=300):
        """
        Creates a Folium map showing the locations of the weather stations in the given subsets.

        Parameters:
        -----------
        - subset_list : list of lists of str
            List of subsets of weather stations, where each subset is a list of station codes.
        - start_date : str, optional
            Start date in the format YYYY-MM-DD, default is None.
        - end_date : str, optional
            End date in the format YYYY-MM-DD, default is None.
        - data_values : bool, optional
            If True, the map markers will display a plot of rainfall data, default is False.
        - csv_file : str, optional
            The name of the CSV file containing the rainfall data, default is 'pr_clog_flags.csv'.
        - min_zoom : int, optional
            The minimum zoom level of the map, default is 8.
        - max_zoom : int, optional
            The maximum zoom level of the map, default is 11.
        - width : int, optional
            The width of the map in pixels, default is 850.
        - height : int, optional
            The height of the map in pixels, default is 850.
        - png_resolution : int, optional
            The resolution of the PNG image if data_values is True, default is 300.

        Returns:
        --------
        - my_map : folium.folium.Map
            A Folium map object showing the locations of the weather stations in the given subsets.
        
        <div align="center">
            <img src="interact.png" alt="Subset Map" width="80%">
        </div>
        """
        # Read the csv file 
        df_rainfall = pd.read_csv(csv_file)
        df_rainfall['Date'] = pd.to_datetime(df_rainfall['Date'])
        df_rainfall= df_rainfall.set_index('Date')
        if start_date and end_date:
            df_rainfall = df_rainfall[start_date:end_date]
        
        # get the stations data to map the centre
        stations = super().get_stations_info()[['code', 'location.longitude', 'location.latitude']] # get the stations data to obtain the distance
        # subset1_stations = list(set([clog.split('_')[0] for clog in subset1])) # ideally the fine stations
        # subset2_stations = list(set([clog.split('_')[0] for clog in subset2])) # clogged stations
        # union_stations = list(set(subset1_stations).union(subset2_stations))
        
        # Changing to use with multiple subsets
        joined = set([j for i in subset_list for j in i])

        # Find the centre from the union of the stations
        stations_df = stations[stations['code'].isin(list(joined))]
        map_center = [stations_df['location.latitude'].mean(), stations_df['location.longitude'].mean()]

        # get the map center
        # map_center = [stations['location.latitude'].mean(), stations['location.longitude'].mean()]

        # create a map centered on a specific location
        my_map = folium.Map(map_center, min_zoom=min_zoom, max_zoom=max_zoom, zoom_start=8, tiles='cartodbpositron', width=width, height=height, png_resolution=png_resolution)
        # add a measure control to the map
        measure_control = MeasureControl(position='topleft', active_color='red', completed_color='red', primary_length_unit='meters')
        my_map.add_child(measure_control)

        # add the markers to the map
        color_markers = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'beige', 'yellow', 'violet', 'brown']
        for _, row in stations.iterrows():
            # loop throught the list subsets and create markers for each subset
            for num, subs in enumerate(subset_list):
                if row['code'] in subs:
                    # Create an IFrame object with the image as the content
                    if data_values:
                        popup_iframe = folium.IFrame(html=f"{self.encode_image(row['code'], df_rainfall)}", width=width//8, height=height//8)
                        folium.Marker([row['location.latitude'], row['location.longitude']], popup=folium.Popup(popup_iframe),  
                                    tooltip=row['code'], icon=folium.Icon(color=color_markers[num])).add_to(my_map)
                    else:
                        folium.Marker([row['location.latitude'], row['location.longitude']], 
                                    tooltip=row['code'], icon=folium.Icon(color=color_markers[num])).add_to(my_map)
        
        # display the map
        return my_map
    


# From the loaded data on the jobs scored, format the data
class transform_data:
    
    def __init__(self, apiKey, apiSecret, api_key):
        super().__init__(apiKey, apiSecret, api_key)

    # transform the station status data
    def transform_station_status(self, station_status, today=datetime.date.today(), transformed_data=True):
        """
        Transforms the station status data into a dictionary with date as the key and online status as the value.

        Parameters:
        ----------
        - station_status (DataFrame): The original DataFrame containing 'id' and 'status' of the stations.
        - today (datetime.date, optional): The date to be used as the index when the job is run. Default is the current date.
        - transformed_data (bool, optional): If True, the data will be transposed and formatted. If False, the original DataFrame will be used with an additional 'Date' column. Default is True.

        Returns:
        -------
        - dict: A dictionary containing the transformed station status data.
            
        Note:
        -----
        - If transformed_data is True:
                The returned dictionary will have the date (today) as the key and the number of stations online for that day as the value.
                Example: {datetime.date(2023, 7, 29): {1: True, 2: False, 3: True}}
            
        - If transformed_data is False:
                The returned dictionary will have each row of the original DataFrame with an additional 'Date' column.
                Example: {0: {'id': 1, 'online': True, 'Date': datetime.date(2023, 7, 29)},
                        1: {'id': 2, 'online': False, 'Date': datetime.date(2023, 7, 29)},
                        2: {'id': 3, 'online': True, 'Date': datetime.date(2023, 7, 29)}}
        """
        # the time of the data to be in the index when the job is run
        if not isinstance(today, datetime.date):
            raise TypeError(f"Expected datetime.date but got {type(today)} instead.")
        # today = datetime.date.today()
        
        if transformed_data:
            # get the station status data
            status_transposed = station_status[['id', 'online']].T
            status_transposed.columns = status_transposed.loc['id']

            # Drop the first row (which was the 'id' column) to make it a proper DataFrame
            status_transposed = status_transposed.drop('id')

            # Convert the index to a datetime index
            status_transposed.index = pd.to_datetime([today])

            # rename the index to date
            status_transposed.index.names = ['Date']
            return status_transposed.to_dict()
        else:
            station_status['Date'] = pd.to_datetime([today])
            return station_status.to_dict()
    



def parse_args():
    parser = argparse.ArgumentParser(description='Locating the different stations')

    parser.add_argument('--address', type=str, required=True, help='Write the address to filter the stations')
    parser.add_argument('--csvfile', default='pr_clog_flags.csv', type=str, help='File to be filtered from default pr_clog_flags.csv')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if args.address or args.csvfile:
        filter.filter_stations(address=args.address, csvfile=args.csvfile)