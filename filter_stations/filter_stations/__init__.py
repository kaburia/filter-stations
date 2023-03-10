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

try:
    # Load json cofig file
    with open('config.json') as f:
        conf = json.load(f)

    apiKey = conf['apiKey']
    apiSecret = conf['apiSecret']
except FileNotFoundError:
    raise("Please create a config.json file with your API key and secret")


# Get the centre point of the address
def getLocation(address):
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
    return requests.get(url).json()[0]['lat'], requests.get(url).json()[0]['lon']

# Get the new radius of the address
def calculate_new_point(lat, lon, distance, bearing):
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
def compute_filter(lat, lon, distance):
    points = []
    g1 = []
    for i in range(0, 360, 45):
        points.append(calculate_new_point(lat, lon, distance, i))
    g1 = [min(p[0] for p in points), min(p[1] for p in points)]
    g2 = [max(p[0] for p in points), max(p[1] for p in points)]
    # print(g1, '\n', g2)
    return g1[0], g1[1], g2[0], g2[1]

# Get the minimum and maximum latitude and longitude of the address


def getStationsInfo(station=None, multipleStations=[]):
    df = []
    reqUrl = "https://datahub.tahmo.org/services/assets/v2/stations"
    headersList = {
    "Accept": "*/*",
    "User-Agent": "Thunder Client (https://www.thunderclient.com)",
    "Authorization": "Basic U2Vuc29yRHhLZW55YTo2R1VYektpI3d2RHZa" 
    }

    payload = ""
    response = requests.request("GET", reqUrl, data=payload,  headers=headersList).json()
    info = pd.json_normalize(response['data']).drop('id', axis=1)
    if station:
        return info[info['code'] == station.upper()]
    elif len(multipleStations) >= 1:
        return info[info['code'].isin(multipleStations)]
    else:
        return info

def filterStations(address, distance, startDate=None, endDate=None, csvfile='KEcheck3.csv'):
    # startdate = dateutil.parser.parse(startDate)
    # enddate = dateutil.parser.parse(endDate)        
    lat, lon = getLocation(address)
    min_lat, min_lon, max_lat, max_lon = compute_filter(float(lat), float(lon), distance)
    # boundingbox = list(map(float, location[0]['boundingbox']))
    # boundingbox_lat = sorted(boundingbox[0:2])
    # boundingbox_lon = sorted(boundingbox[2:])
    stations = getStationsInfo()
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
def filterStationsList(address, distance=100):
    return list(set([i.split('_')[0] for i in filterStations(f'{address}', distance).columns if i.split('_')[-1] != 'clogFlag']))

# A dictionary of neighbouring stations
def k_neighbours(station, number=5):
    '''
    Pass in a station - return the longitude and latitude from get stations info
    '''
    lon, lat = getStationsInfo(station)[['location.longitude', 'location.latitude']].values[0]
    infostations = getStationsInfo()
    infostations['distance'] = infostations.apply(lambda row: hs.haversine((lat, lon), (row['location.latitude'], row['location.longitude'])), axis=1)
    infostations = infostations.sort_values('distance')
    return  dict(infostations[['code', 'distance']].head(number).values[1:])


def draw_map(map_center):
    #  retrieve the stations data
    stations = getStationsInfo()[['code', 'location.longitude', 'location.latitude']]
    # create a map centered on a specific location

    my_map = folium.Map(location=map_center, min_zoom=5, max_zoom=30, zoom_start=12, tiles='cartodbpositron')
    for _, row in stations.iterrows():
        folium.Marker([row['location.latitude'], row['location.longitude']], tooltip=row['code']).add_to(my_map)


    # fit the map bounds to the markers' locations
    marker_locations = [[row['location.latitude'], row['location.longitude']] for _, row in stations.iterrows()]
    my_map.fit_bounds(marker_locations)

    # display the map
    return my_map


def create_animation(data, valid_sensors, day=100, T=10, interval=500):
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
def plot_station(ws, df_rainfall):
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
def encode_image(ws, df_rainfall):
    figure = plot_station(ws, df_rainfall)
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



def get_map(subset1, subset2, start_date=None, end_date=None, data_values=False, csv_file='KEcheck3.csv', min_zoom=8, max_zoom=11):
    # Read the csv file 
    df_rainfall = pd.read_csv(csv_file)
    df_rainfall['Date'] = pd.to_datetime(df_rainfall['Date'])
    df_rainfall= df_rainfall.set_index('Date')
    if start_date and end_date:
        df_rainfall = df_rainfall[start_date:end_date]
    # get the stations data
    stations = getStationsInfo()[['code', 'location.longitude', 'location.latitude']]
    subset1_stations = list(set([clog.split('_')[0] for clog in subset1])) # ideally the fine stations
    subset2_stations = list(set([clog.split('_')[0] for clog in subset2])) # clogged stations
    union_stations = list(set(subset1_stations).union(subset2_stations))

    # Find the centre from the union of the stations
    stations_df = stations[stations['code'].isin(union_stations)]
    map_center = [stations_df['location.latitude'].mean(), stations_df['location.longitude'].mean()]

    # get the map center
    # map_center = [stations['location.latitude'].mean(), stations['location.longitude'].mean()]

    # create a map centered on a specific location
    my_map = folium.Map(map_center, min_zoom=min_zoom, max_zoom=max_zoom, zoom_start=8, tiles='cartodbpositron')

    # add the markers to the map
    for _, row in stations.iterrows():
        if row['code'] in subset1_stations:
            # Create an IFrame object with the image as the content
            if data_values:
                popup_iframe = folium.IFrame(html=f"{encode_image(row['code'], df_rainfall)}", width=400, height=300)
                folium.Marker([row['location.latitude'], row['location.longitude']], popup=folium.Popup(popup_iframe),  
                            tooltip=row['code']).add_to(my_map)
            else:
                folium.Marker([row['location.latitude'], row['location.longitude']], 
                            tooltip=row['code']).add_to(my_map)                
        if row['code'] in subset2_stations:
            if data_values:
                popup_iframe = folium.IFrame(html=f"{encode_image(row['code'], df_rainfall)}", width=400, height=300)
                folium.Marker([row['location.latitude'], row['location.longitude']], popup=folium.Popup(popup_iframe),  
                            tooltip=row['code'], icon=folium.Icon(color='red', icon='info-sign')).add_to(my_map)
            else:
                folium.Marker([row['location.latitude'], row['location.longitude']],  
                            tooltip=row['code'], icon=folium.Icon(color='red', icon='info-sign')).add_to(my_map)
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
        filterStations(address=args.address, csvfile=args.csvfile)