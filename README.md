
## Documentation
You can find the documentation for the project by following this link<br>
https://filter-stations.netlify.app/


Getting Started
---------------
All methods require an API key and secret, which can be obtained by contacting TAHMO. <br>
- The ```retreive_data``` class is used to retrieve data from the TAHMO API endpoints.<br> 
- The ```Filter``` class is used to filter weather stations data based on things like distance and region.<br>
- The ```pipeline``` class is used to create a pipeline of filters to apply to weather stations based on how they correlate with water level data.<br>
- The ```Interactive_maps``` class is used to plot weather stations on an interactive map.<br>

For instructions on shedding weather stations based on your water level data and gauging station coordinates, please refer to the [water_level_pipeline.md](https://github.com/kaburia/filter-stations/blob/v0.5.1/water_level_pipeline.md) file.