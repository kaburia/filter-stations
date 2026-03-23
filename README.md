## Documentation
You can find the documentation for the project by following this link<br>
https://filter-stations.readthedocs.io/en/latest/

Getting Started
---------------
All methods require an API key and secret, which can be obtained by contacting `dsail-info@dkut.ac.ke`. <br>
 
 - ```AuthManager```: The new secure gateway. Log in once with your credentials to automatically unlock access to all downstream data loaders
- ```RainLoader```: class is used to get our DSAIL unified weather dataset from HuggingFace (See the documentation for more information on this) <br>
- ```MediumForecaster```: Connects to the Google Weather API to pull 1-to-10 day forecasts (both hourly and daily)
- ```SeasonalForecaster```: Connects to Open-Meteo API to pull seasonal forecasts.
- The ```RetrieveData``` class is used to retrieve data from the TAHMO API endpoints.<br>
- The ```Kieni``` class is used to get weather data for stations 100km around Kieni from the central point.with water level data.<br>

### Example
```python
from filter_stations import AuthManager, MediumForecaster, RetrieveData, SeasonalForecaster

# 1. Authenticate once securely
auth = AuthManager(email="your_email@domain.com", password="your_password")

# 2. Initialize any tool you need seamlessly
weather = MediumForecaster(auth_session=auth)
tahmo = RetrieveData(auth_session=auth)
seasonal = SeasonalForecaster(auth_session=auth)

# 3. Pull your Medium Range weather forecast data 
nyeri_forecast = weather.get_weather_api_forecast(lat=-0.41, lon=36.95, days=7)

# Seasonal Forecasting Example

# 4. See what is available
print(seasonal.list_all_models())

# 5. Pull 6 months of ECMWF AIFS and Standard IFS data simultaneously
data = seasonal.get_forecast(
    lat=-0.41,
    lon=36.95,
    models=["ecmwf_seasonal_ensemble_mean_seamless"], 
    forecast_months=6,
    daily=['temperature_2m_max',
            'temperature_2m_mean',
            'temperature_2m_min',
            'precipitation_sum',
            'rain_sum',
            'pressure_msl_min',
            'pressure_msl_mean',
            'relative_humidity_2m_min',
            'relative_humidity_2m_mean',
            'relative_humidity_2m_max',
            'dew_point_2m_max',
            'dew_point_2m_mean',
            'dew_point_2m_min',
            'surface_pressure_min',
            'surface_pressure_mean',
            'surface_pressure_max',
            'cloud_cover_min',
            'cloud_cover_mean',
            'cloud_cover_max' ]
)

# 6. View the AIFS DataFrame (first model in the list)
aifs_df = data[0]["daily_df"]
print(aifs_df)

seasonal.list_api_variables()
```

<!-- - The ```Interactive_maps``` class is used to plot weather stations on an interactive map.<br>
- The ```Water_level``` class is used to retrieve water level data and coordinates of gauging stations.<br> -->

<!-- For instructions on shedding weather stations based on your water level data and gauging station coordinates, please refer to the [water_level_pipeline.md](https://github.com/kaburia/filter-stations/blob/main/water_level_pipeline.md) file. -->
<br>

To get started on the module test it out on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KUUtvozBTePyezc1i5hhDuFFSWLzEcXH?usp=sharing)


For earlier versions `<= v0.6.2` use the link below for documentation <br>

https://filter-stations.netlify.app/

## Citations

If you use this package in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{filter-stations,
  author = {Austin Kaburia},
  title = {filter-stations},
  year = {2024},
  publisher = {Python Package Index},
  journal = {PyPI},
  howpublished = {\url{https://pypi.org/project/filter-stations/}},
}
