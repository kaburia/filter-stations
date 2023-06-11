import requests
import urllib.parse
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


# Constants
API_BASE_URL = 'https://datahub.tahmo.org'
API_MAX_PERIOD = '1Y'

endpoints = {'VARIABLES': 'services/assets/v2/variables', # 28 different variables
             'STATION_INFO': 'services/assets/v2/stations',
             'WEATHER_DATA': 'services/measurements/v2/stations', # Configured before requesting
             'DATA_COMPLETE': 'custom/sensordx/latestmeasurements',
             'STATION_STATUS': 'custom/stations/status'}

# authentication class


# Get data class
class retreive_data:
    # initialize the class
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
        print(f'API request: {endpoint}')
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

        """
        # Make API request and convert response to DataFrame
        response = self.__request(endpoints['STATION_INFO'], {'sort':'code'})
        info = pd.json_normalize(response['data']).drop('id', axis=1)
        
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
    
    def raw_measurements(self, station, startDate=None, endDate=None, variables=None):
        return self.get_measurements(station, startDate=startDate, endDate=endDate, variables=variables, dataset='raw')
    
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
        reqUrl = "https://tahmorqctest.eu-de.mybluemix.net/api/models" # endpoint
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
        
    
    def aggregate_variables(self, dataframe):
        """
        Aggregates a pandas DataFrame of weather variables by summing values across each day.

        Parameters:
        -----------
        - dataframe (pandas.DataFrame): DataFrame containing weather variable data.

        Returns:
        -----------
        - pandas.DataFrame: DataFrame containing aggregated weather variable data, summed by day.
        """
        # Reset index and rename columns
        dataframe = dataframe.reset_index()
        dataframe.rename(columns={'index':'Date'}, inplace=True)
        
        # Group by date and sum values
        return dataframe.groupby(pd.Grouper(key='Date', axis=0, freq='1D')).sum()
    
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
    def get_measurements(self, station, startDate=None, endDate=None, variables=None, dataset='controlled', aggregate=False, quality_flags=False):
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
            if quality_flags:
                # cols = [col for col in df.columns if col.split('_')[-1] == 'Q_FLAG']
                # print(cols)
                # df = df[cols]
                # df = df[[f'{station}_Q_Flag']]
                if aggregate:
                    return self.aggregate_qualityflags(df)
                else:
                    return df
            else:
                if aggregate:
                    return self.aggregate_variables(df)
                else:
                    return df
    
    # retrieve data from multiple at a time
    def multiple_measurements(self, stations_list, csv_file, startDate, endDate, variables, dataset='controlled'):
        """
        Retrieves measurements for multiple stations and saves the aggregated data to a CSV file.

        Parameters:
        -----------
        - stations_list (list): A list of strings containing the names of the stations to retrieve data from.
        - csv_file (str): The name of the CSV file to save the data to.
        - startDate (str): The start date for the measurements, in the format 'yyyy-mm-dd'.
        - endDate (str): The end date for the measurements, in the format 'yyyy-mm-dd'.
        - variables (list): A list of strings containing the names of the variables to retrieve.
        - dataset (str): The name of the dataset to retrieve the data from. Default is 'controlled'.

        Returns:
        -----------
        - df (pandas.DataFrame): A DataFrame containing the aggregated data for all stations.

        Raises:

            ValueError: If stations_list is not a list.
        """
        error_dict = dict()
        if isinstance(stations_list, list):
            df_stats = []
            
            for station in stations_list:
                print(stations_list.index(station))
                print(f'Retrieving data for station: {station}')
                try:
                    data = self.get_measurements(station, startDate, endDate, variables)
                    agg_data = self.aggregate_variables(data)
                    df_stats.append(agg_data)
                except Exception as e:
                    error_dict[station] = f'{e}'
            
            with open("Errors.json", "w") as outfile:
                json.dump(error_dict, outfile, indent=4)
            
            if len(df_stats) > 0:
                df = pd.concat(df_stats, axis=1)
                df.to_csv(f'{csv_file}.csv')
                return df

        
        else:
            raise ValueError('Pass in a list')
        
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

        

# Move the functions to a class
class Filter(retreive_data):
    # inherit from retrieve_data class
    def __init__(self, apiKey, apiSecret):
        super().__init__(apiKey, apiSecret)
    
    def get_stations_info(self, station=None, multipleStations=[], countrycode=None):
        return super().get_stations_info(station, multipleStations, countrycode)
        
    # Get the centre point of the address
    def centre_point(self, address):
        """
        This method retrieves the latitude and longitude coordinates of a given address using the Nominatim API.
        
        Parameters:
        -----------
        - address : str
            The address of the location you want to retrieve the coordinates for.
            
        Returns:
        --------
        - Tuple (float, float)
            The latitude and longitude coordinates of the location.
        """
        url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
        return requests.get(url).json()[0]['lat'], requests.get(url).json()[0]['lon']

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


    def filter_stations(self, address, distance, startDate=None, endDate=None, csvfile='KEcheck3.csv'):
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
        lat, lon = self.centre_point(address)
        min_lat, min_lon, max_lat, max_lon = self.compute_filter(float(lat), float(lon), distance)
        stations = super().get_stations_info()
        bounds = list(stations['code'][(stations['location.longitude'] >= min_lon)
                                        & (stations['location.longitude'] <= max_lon)
                                        & (stations['location.latitude'] >= min_lat)
                                            & (stations['location.latitude'] <= max_lat)])
        
        # read the csv file
        ke_chec = pd.read_csv(csvfile)
        ke_chec.Date = ke_chec.Date.astype('datetime64')
        # print(ke_chec.info())

        # ke_chec = ke_chec.set_index('Date')
        if startDate and endDate:
            startdate = dateutil.parser.parse(startDate)
            enddate = dateutil.parser.parse(endDate)
            begin = ke_chec['Date'][ke_chec['Date'] == startdate].index.to_numpy()[0]
            end = ke_chec['Date'][ke_chec['Date'] == enddate].index.to_numpy()[0]
            ke_chec = ke_chec.iloc[begin:end+1]
            ke_chec = ke_chec.set_index('Date')

            return ke_chec[[col for bbox in bounds for col in ke_chec if bbox in col]]
        else:
            ke_chec = ke_chec.set_index('Date')
            return ke_chec[[col for bbox in bounds for col in ke_chec if bbox in col]]


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
    def __init__(self, apiKey, apiSecret):
        super().__init__(apiKey, apiSecret)

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
        - data (DataFrame): A pandas DataFrame containing station data defaults to none reads KEcheck3 if none.
        - sensors (list): A list of valid sensor names.
        - day (int): The starting day of the animation (default is 100).
        - T (int): The range of days for the animation (default is 10).
        - interval (int): The interval between frames in milliseconds (default is 500).

        Returns:
        -----------
        - HTML: An HTML object containing the animation.
        '''
        if not data:
            data = pd.read_csv('KEcheck3.csv')
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



    def get_map(self, subset_list, start_date=None, end_date=None, data_values=False, csv_file='KEcheck3.csv', min_zoom=8, max_zoom=11, width=2000, height=2000, png_resolution=300):
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
            The name of the CSV file containing the rainfall data, default is 'KEcheck3.csv'.
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

def parse_args():
    parser = argparse.ArgumentParser(description='Locating the different stations')

    parser.add_argument('--address', type=str, required=True, help='Write the address to filter the stations')
    parser.add_argument('--csvfile', default='KEcheck3.csv', type=str, help='File to be filtered from default KEcheck3.csv')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if args.address or args.csvfile:
        filter.filter_stations(address=args.address, csvfile=args.csvfile)