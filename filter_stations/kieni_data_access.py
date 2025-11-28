import pandas as pd
import requests


class Kieni:
    # uses basic authentication
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
 
    # get data from the kieni API
    def kieni_weather_data(self, start_date=None, end_date=None, variable=None,
                        method='sum', freq='1D'):
        """
        Retrieves weather data from the Kieni API endpoint and returns it as a pandas DataFrame after processing.

        Parameters
        ----------
        start_date : str, optional
            The start date for retrieving weather data in 'YYYY-MM-DD' format.
            Defaults to None (returns from the beginning of the data).
        end_date : str, optional
            The end date for retrieving weather data in 'YYYY-MM-DD' format.
            Defaults to None (returns to the end of the data).
        variable : str, optional
            The weather variable to retrieve (same as the weather shortcodes by TAHMO e.g., 'pr', 'ap', 'rh').
        method : str, optional
            The aggregation method to apply to the data ('sum', 'mean', 'min', 'max' and custom functions).
            Defaults to 'sum'.
        freq : str, optional
            The frequency for data aggregation (e.g., '1D' for daily, '1H' for hourly). Defaults to '1D'.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the weather data for the specified parameters, with columns containing NaN values dropped.

        Usage
        -----
        To retrieve daily rainfall data from January 1, 2024, to January 31, 2024::

            # Instantiate the Kieni class
            api_key, api_secret = '', '' # Request DSAIL for the API key and secret
            kieni = Kieni(api_key, api_secret)

            kieni_weather_data = kieni.kieni_weather_data(
                start_date='2024-01-01', 
                end_date='2024-01-31', 
                variable='pr', 
                freq='1D', 
                method='sum'
            )

        To retrieve hourly temperature data from February 1, 2024, to February 7, 2024::

            kieni_weather_data = kieni.kieni_weather_data(
                start_date='2024-02-01', 
                end_date='2024-02-07', 
                variable='te', 
                method='mean', 
                freq='1H'
            )
        """
        # Make the request
        reqUrl = f"https://us-central1-tahmo-quality-control.cloudfunctions.net/kieni-API?start_date={start_date}&end_date={end_date}&variable={variable}&freq={freq}&method={method}"
        apiRequest = requests.get(reqUrl, auth=requests.auth.HTTPBasicAuth(self.api_key, self.api_secret))
        response = apiRequest.json()
        # convert the response to a pandas dataframe
        data = pd.DataFrame(response)
        # drop all columns that are filled with NaN
        data.dropna(axis=1, how='all', inplace=True)
        # convert index to datetime
        data.index = pd.to_datetime(data.index)
        return data
