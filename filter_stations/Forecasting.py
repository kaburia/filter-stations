# methods for loading and forecasting station data
'''
Medium Range forecasting will be extracted from Google Weather API
Seasonal Forecasting will be extracted from IRI
'''

import requests
import warnings

class MediumForecaster:
    """
    Handles medium-range (1 to 10 days) weather forecasting by securely 
    retrieving credentials and querying the Google Weather API.
    """
    def __init__(self, auth_session):
        warnings.filterwarnings('ignore')
        
        # 1. Ask the Gatekeeper for the Google Weather API key
        # We pass context in the 'details' dictionary for your Cloud Storage audit trail
        credentials = auth_session.request_access(
            service_name='weatherapi', 
            details={"action": "initialize_medium_forecaster"}
        )
        
        # 2. Store the retrieved key locally for the session
        self.api_key = credentials.get('key')
        
        if not self.api_key:
            raise ValueError("Failed to retrieve the Weather API key from the gatekeeper.")

    def get_weather_api_forecast(self, lat, lon, days=None, hours=None):
        """
        Retrieves raw forecasting data from the Google Weather API.
        Automatically handles pagination.
        You must specify either 'days' (max 10) OR 'hours' (max 240).
        """
        # 1. Input Validation
        if days is not None and hours is not None:
            raise ValueError("Please specify either 'days' or 'hours', not both.")
        if days is None and hours is None:
            days = 10  # Default behavior if nothing is passed
            
        # 2. Configure the API Request based on the user's choice
        if days is not None:
            if days < 1 or days > 10:
                raise ValueError("Google Weather API supports between 1 and 10 days.")
            endpoint = "forecast/days:lookup"
            time_param = f"&days={days}"
            data_key = "forecastDays"  # The key Google uses in the JSON response
            print(f"Fetching {days}-day forecast for coordinates: {lat}, {lon}...")
            
        if hours is not None:
            if hours < 1 or hours > 240:
                raise ValueError("Google Weather API supports between 1 and 240 hours.")
            endpoint = "forecast/hours:lookup"
            time_param = f"&hours={hours}"
            data_key = "forecastHours" # The key Google uses in the JSON response
            print(f"Fetching {hours}-hour forecast for coordinates: {lat}, {lon}...")

        # 3. Setup Pagination Variables
        all_forecast_data = []
        time_zone_info = None 
        
        base_url = (
            f"https://weather.googleapis.com/v1/{endpoint}"
            f"?key={self.api_key}"
            f"&location.latitude={lat}"
            f"&location.longitude={lon}"
            f"{time_param}"
        )
        
        next_page_token = None
        
        # 4. The Pagination Loop
        while True:
            current_url = base_url
            if next_page_token:
                current_url += f"&pageToken={next_page_token}"
                
            response = requests.get(current_url)
            
            if response.status_code != 200:
                raise ConnectionError(f"Weather API request failed: {response.text}")
                
            data = response.json()
            
            # Extract the data using the dynamic data_key (either days or hours)
            if data_key in data:
                all_forecast_data.extend(data[data_key])
                
            # Capture timezone once
            if not time_zone_info and 'timeZone' in data:
                time_zone_info = data['timeZone']
                
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
                
        # 5. Reconstruct and Return
        unified_response = {
            data_key: all_forecast_data,
            "timeZone": time_zone_info
        }
        
        return unified_response