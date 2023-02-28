import requests
import urllib.parse
import pandas as pd
import argparse
import dateutil.parser
import math

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

def parse_args():
    parser = argparse.ArgumentParser(description='Locating the different stations')

    parser.add_argument('--address', type=str, required=True, help='Write the address to filter the stations')
    parser.add_argument('--csvfile', default='KEcheck3.csv', type=str, help='File to be filtered from default KEcheck3.csv')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if args.address or args.csvfile:
        filterStations(address=args.address, csvfile=args.csvfile)