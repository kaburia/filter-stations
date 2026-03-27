# methods for loading and forecasting station data
'''
Medium Range forecasting will be extracted from Google Weather API
Seasonal Forecasting will be extracted from IRI
'''

import requests
import warnings
import openmeteo_requests
import requests_cache
import pandas as pd
import json
import os
from retry_requests import retry

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

        :param lat: Latitude of the location.
        :type lat: float
        :param lon: Longitude of the location.
        :type lon: float
        :param days: Number of days to forecast (1 to 10). Defaults to None.
        :type days: int, optional
        :param hours: Number of hours to forecast (1 to 240). Defaults to None.
        :type hours: int, optional
        :return: A dictionary containing the forecast data and timezone information.
        :rtype: dict
        :raises ValueError: If both days and hours are specified, or if values are out of range.
        :raises ConnectionError: If the API request fails.
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
    


class SeasonalForecaster:
    def __init__(self, auth_session=None, catalog_path=None):
        """
        Initializes the seasonal forecaster and loads the complete variable/model catalog.
        """
        self.auth = auth_session
        self.url = "https://seasonal-api.open-meteo.com/v1/seasonal"
        
        if catalog_path is None:
            catalog_path = os.path.join(os.path.dirname(__file__), "openmeteo_catalog.json")
        
        # Load the external catalog
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Missing configuration file: {catalog_path}.")
            
        with open(catalog_path, "r") as f:
            self.catalog = json.load(f)

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

    # --- Catalog Helper Methods ---
    def list_api_variables(self):
        """
        Returns the pre-formatted variable strings to pass directly to the API.

        :return: A dictionary of API variables available in the catalog.
        :rtype: dict
        """
        return self.catalog.get("api_variables", {})

    def list_all_models(self):
        """
        Returns every model identifier defined in the Open-Meteo SDK.

        :return: A list of model names sorted alphabetically.
        :rtype: list[str]
        """
        return sorted(self.catalog.get("fbs_reference", {}).get("models", []))

    def list_base_variables(self):
        """Returns the raw physical variables defined in the Open-Meteo FlatBuffers."""
        return self.catalog.get("fbs_reference", {}).get("variables", [])

    def list_aggregations(self):
        """Returns the raw FlatBuffer aggregation types supported by Open-Meteo."""
        return self.catalog.get("fbs_reference", {}).get("aggregations", [])
        
    def list_units(self):
        """Returns all supported measurement units."""
        return self.catalog.get("fbs_reference", {}).get("units", [])

    def list_probabilities(self):
        """Returns all supported probability thresholds."""
        return self.catalog.get("fbs_reference", {}).get("probabilities", [])

    # --- Core API Methods
    def get_forecast(self, lat, lon, models=None, forecast_months=3, 
                     hourly=None, daily=None, weekly=None, monthly=None):
        """
        Fetches seasonal forecast data across any requested temporal resolution.

        :param lat: Latitude of the location.
        :type lat: float
        :param lon: Longitude of the location.
        :type lon: float
        :param models: List of weather models to use for the forecast. Defaults to None.
        :type models: list[str], optional
        :param forecast_months: Number of months to forecast. Defaults to 3.
        :type forecast_months: int, optional
        :param hourly: List of hourly variables to request.
        :type hourly: list[str], optional
        :param daily: List of daily variables to request.
        :type daily: list[str], optional
        :param weekly: List of weekly variables to request.
        :type weekly: list[str], optional
        :param monthly: List of monthly variables to request.
        :type monthly: list[str], optional
        :return: A list of dictionaries, where each dictionary represents the forecast data for a specific model 
                 and location. Keys include 'latitude', 'longitude', 'model_index', and DataFrames for requested timeframes.
        :rtype: list[dict]
        :raises ValueError: If no variables (hourly, daily, weekly, or monthly) are requested.
        """
        hourly, daily = hourly or [], daily or []
        weekly, monthly = weekly or [], monthly or []
        
        if not any([hourly, daily, weekly, monthly]):
            raise ValueError("You must request at least one variable array (hourly, daily, weekly, or monthly).")
            
        params = {
            "latitude": lat, "longitude": lon, "forecast_months": forecast_months,
        }
        
        if models: params["models"] = models
        if hourly: params["hourly"] = hourly
        if daily: params["daily"] = daily
        if weekly: params["weekly"] = weekly
        if monthly: params["monthly"] = monthly

        responses = self.client.weather_api(self.url, params=params)
        
        results = []
        for i, response in enumerate(responses):
            location_data = {
                "latitude": response.Latitude(),
                "longitude": response.Longitude(),
                "model_index": i
            }
            
            # Pass the timeframe so the parser knows how to handle the dates
            if hourly: location_data["hourly_df"] = self._parse_flatbuffer(response.Hourly(), hourly, "standard")
            if daily: location_data["daily_df"] = self._parse_flatbuffer(response.Daily(), daily, "standard")
            if weekly: location_data["weekly_df"] = self._parse_flatbuffer(response.Weekly(), weekly, "standard")
            if monthly: location_data["monthly_df"] = self._parse_flatbuffer(response.Monthly(), monthly, "monthly")
                
            results.append(location_data)
            
        return results

    # def calculate_ensemble_mean(self, df, variable_name):
    #     """Calculates the mean across all ensemble members for a specific variable."""
    #     member_cols = [col for col in df.columns if col.startswith(f"{variable_name}_member")]
    #     if not member_cols: return df # Skip if it's already an aggregated/anomaly variable
    #     df[f"{variable_name}_ensemble_mean"] = df[member_cols].mean(axis=1)
    #     return df

    def _parse_flatbuffer(self, data_block, requested_vars, timeframe):
        """
        Smartly extracts data, handling Int64 arrays (sunrise/sunset) 
        and missing Ensemble Members (weekly/monthly anomalies).
        """
        # 1. Handle time index formatting
        if timeframe == "monthly":
            date_range = pd.date_range(
                start=f"{data_block.Year()}-{data_block.Month()}-01",
                periods=data_block.Count(), freq="MS"
            )
        else:
            date_range = pd.date_range(
                start=pd.to_datetime(data_block.Time(), unit="s", utc=True),
                end=pd.to_datetime(data_block.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=data_block.Interval()), inclusive="left"
            )
        
        parsed_data = {"date": date_range}
        
        # 2. Determine member chunking
        num_requested_vars = len(requested_vars)
        total_variables_returned = data_block.VariablesLength()
        members_per_var = total_variables_returned // num_requested_vars

        var_index = 0
        for var_name in requested_vars:
            for _ in range(members_per_var):
                variable = data_block.Variables(var_index)
                
                # 3. Handle Column Naming (Catch missing Ensemble Members in Weekly/Monthly)
                try:
                    member_id = variable.EnsembleMember()
                    col_name = f"{var_name}_member{member_id}"
                except Exception:
                    # If it has no members (e.g., precipitation_anomaly), just use the var name
                    col_name = var_name if members_per_var == 1 else f"{var_name}_{_}"

                # 4. Handle Int64 formatting (Catch Sunrise/Sunset Unix timestamps)
                if var_name in ["sunrise", "sunset"]:
                    parsed_data[col_name] = variable.ValuesInt64AsNumpy()
                else:
                    parsed_data[col_name] = variable.ValuesAsNumpy()
                    
                var_index += 1
                
        return pd.DataFrame(data=parsed_data)