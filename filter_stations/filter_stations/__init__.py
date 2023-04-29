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

# try:
#     # Load json cofig file
#     with open('config.json') as f:
#         conf = json.load(f)

#     apiKey = conf['apiKey']
#     apiSecret = conf['apiSecret']
# except FileNotFoundError:
#     raise("Please create a config.json file with your API key and secret")
# Constants
API_BASE_URL = 'https://datahub.tahmo.org'
API_MAX_PERIOD = '365D'

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
        response = self.__request(endpoints['STATION_INFO'], {'sort':'code'})
        info = pd.json_normalize(response['data']).drop('id', axis=1)
        if station:
            return info[info['code'] == station.upper()]
        elif len(multipleStations) >= 1:
            return info[info['code'].isin(multipleStations)]
        elif countrycode:
            info = info[info['location.countrycode'] == f'{countrycode.upper()}']
            return info.drop(labels=info['code'][info.code.str.contains('TH')].index, axis=0)
        else:
            return info
        
    # retrieve available variables
    def get_variables(self):
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
    
    # A dictionary of neighbouring stations
    def k_neighbours(self, station, number=5):
        '''
        Pass in a station - return the longitude and latitude from get stations info
        '''
        lon, lat = self.get_stations_info(station)[['location.longitude', 'location.latitude']].values[0]
        infostations = self.get_stations_info()
        infostations['distance'] = infostations.apply(lambda row: hs.haversine((lat, lon), (row['location.latitude'], row['location.longitude'])), axis=1)
        infostations = infostations.sort_values('distance')
        return  dict(infostations[['code', 'distance']].head(number).values[1:])
    
    # Pure aggregates
    def aggregate_variables(self, dataframe):
        dataframe = dataframe.reset_index()
        dataframe.rename(columns = {'index':'Date'}, inplace = True)
        return dataframe.groupby(pd.Grouper(key='Date', axis=0, 
                        freq='1D')).sum()
    
    # Get the variables only
    def get_measurements(self, station, startDate=None, endDate=None, variables=None, dataset='controlled', aggregate=False):
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
                    values = list(map(lambda x: x[1], seriesHolder[shortcode]))
                    serie = pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=np.float64)

                    if len(values) > 0:
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
            if aggregate:
                return self.aggregate_variables(df)
            else:
                return df
    
    # retrieve data from multiple at a time
    def multiple_measurements(self, stations_list, csv_file, startDate, endDate, variables, dataset='controlled'):
        error_dict = dict()
        if isinstance(stations_list, list):
            df_stats = []
            
            for station in stations_list:
                print(stations_list.index(station))
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



# Move the functions to a class
class Filter(retreive_data):
    # inherit from retrieve_data class
    def __init__(self, apiKey, apiSecret):
        super().__init__(apiKey, apiSecret)
    
    def get_stations_info(self, station=None, multipleStations=[], countrycode=None):
        return super().get_stations_info(station, multipleStations, countrycode)
        
    # Get the centre point of the address
    def getLocation(self, address):
        url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
        return requests.get(url).json()[0]['lat'], requests.get(url).json()[0]['lon']

    # Get the new radius of the address
    def calculate_new_point(self, lat, lon, distance, bearing):
        """
        Calculate a new point given a starting point, distance, and bearing.
        
        :param lat: starting latitude in degrees
        :param lon: starting longitude in degrees
        :param distance: distance to move in meters
        :param bearing: bearing to move in degrees (0 is north)
        :return: tuple containing the new latitude and longitude in degrees
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
        points = []
        g1 = []
        for i in range(0, 360, 45):
            points.append(self.calculate_new_point(lat, lon, distance, i))
        g1 = [min(p[0] for p in points), min(p[1] for p in points)]
        g2 = [max(p[0] for p in points), max(p[1] for p in points)]
        # print(g1, '\n', g2)
        return g1[0], g1[1], g2[0], g2[1]

    # Get the minimum and maximum latitude and longitude of the address


    def filterStations(self, address, distance, startDate=None, endDate=None, csvfile='KEcheck3.csv'):     
        lat, lon = self.getLocation(address)
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
    def filterStationsList(self, address, distance=100):
        return list(set([i.split('_')[0] for i in self.filterStations(f'{address}', distance).columns if i.split('_')[-1] != 'clogFlag']))

    

# A different class for visualisations
class Interactive_maps(retreive_data):
    # inherit from retrieve_data class
    def __init__(self, apiKey, apiSecret):
        super().__init__(apiKey, apiSecret)

    def draw_map(self, map_center):
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


    def create_animation(self, data, valid_sensors, day=100, T=10, interval=500):
        '''
        T days giving the range 
        valid sensors
        '''
        
        data1 = np.array(data[valid_sensors].iloc[day:day+1]).T
        data1 /= np.max(data1)
        # Create figure and axis
        fig, ax = plt.subplots()

        # Define function to update matrix data
        def update(i):
            # Generate new data
            new_data =  np.array(data[valid_sensors].iloc[int(day):int(day+1+i)]).T
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


    '''
    1. Get the stations information
    2. From the points given get the map center
    3. Get a max and min zoom for the map
    4. Place the data in a map with a certain colour
    5. a different marker for a different subset of the data
    6. Pass an argument for the start and end date
    7. Generate plots for each station for the given date range and add them to the map as a popup on click
    8. Create a slider to change the day on the plots 
    '''
    def plot_station(self, ws, df_rainfall):
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



    def get_map(self, subset_list, start_date=None, end_date=None, data_values=False, csv_file='KEcheck3.csv', min_zoom=8, max_zoom=11, width=850, height=850, png_resolution=300):
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
                        popup_iframe = folium.IFrame(html=f"{self.encode_image(row['code'], df_rainfall)}", width=width//2, height=height//2)
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
        filter.filterStations(address=args.address, csvfile=args.csvfile)