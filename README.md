## Documentation
https://filter-stations.netlify.app/

## Installation
```
pip install filter-stations
```

## Water Level Pipeline 
- A series of functions to be added to the filter-stations module in pypi to evalute which TAHMO stations to use that corroborates with the water level
- All begins with the coordinates of the gauging station(location of the monitoring sensor)


```python
import os
from pathlib import Path
import haversine as hs
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import warnings
import dateutil.parser
warnings.filterwarnings('ignore')

# config_path
config_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'config.json')
```


```python
from filter_stations import retreive_data, Interactive_maps, Filter, pipeline
import json
# Authentication
with open(config_path) as f:
    conf = json.load(f)

apiKey = conf['apiKey']
apiSecret = conf['apiSecret']
map_api_key = conf['map_api_key']
fs = retreive_data(apiKey, apiSecret, map_api_key)
pipe = pipeline(apiKey, apiSecret, map_api_key)
maps = Interactive_maps(apiKey, apiSecret, map_api_key)
```

### Loading data
Load the water level data from the github repository[Link here] <br>
Load the TAHMO station data from the [Link here] <br>


```python
# muringato 
muringato_loc = [-0.406689, 36.96301]  
# ewaso 
ewaso_loc = [0.026833, 36.914637]

# Weather stations data
weather_stations_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data', 'stations_precipitation.csv'))

''' The water level data '''
# muringato data sensor 2 2021
muringato_data_s2_2021 = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data', 'water_data_2021', 'muringato-sensor2.csv')

# muringato data sensor 2 2022
muringato_data_s2_2022 = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data', 'water_data_2021', 'muringato-sensor2-2022.csv')

# muringato data sensor 6 2021
muringato_data_s6_2021 = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data', 'water_data_2021', 'muringato-sensor6.csv')

# muringato data sensor 6 2022
muringato_data_s6_2022 = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data', 'water_data_2021', 'muringato-sensor6-2022.csv')


# ewaso data sensor 2020 convert the time column to datetime
ewaso_data_2020 = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data', 'water-level-data-ewaso', '1E2020.csv')

# ewaso data sensor 2022
ewaso_data_2022 = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data', 'water-level-data-ewaso', '1E2022.csv')

weather_stations_data.Date = weather_stations_data.Date.astype('datetime64[ns]')
weather_stations_data.set_index('Date', inplace=True)

```

To format water level it needs to have a time column and water level column the names can be different but the order must be that


```python
# handle the water level data
def format_water_level(water_level_data_path):
    # data needs to be in the format time, data/water_level or whatever the column is called
    water_level_data = pd.read_csv(water_level_data_path)
    # rename the first column to time
    water_level_data.rename(columns={water_level_data.columns[0]: 'time'}, inplace=True)
    # convert the time column to datetime
    water_level_data.time = pd.to_datetime([dateutil.parser.parse(i).strftime('%d-%m-%Y') for i in water_level_data['time']])
    water_level_data.time = water_level_data.time.astype('datetime64[ns]')
    # rename the column to water_level
    water_level_data.rename(columns={water_level_data.columns[1]: 'water_level'}, inplace=True)
    # set the time column as the index
    water_level_data.set_index('time', inplace=True)
    return water_level_data
```


```python
muringato_data_s2_2021 = format_water_level(muringato_data_s2_2021)
muringato_data_s2_2022 = format_water_level(muringato_data_s2_2022)
muringato_data_s6_2021 = format_water_level(muringato_data_s6_2021)
muringato_data_s6_2022 = format_water_level(muringato_data_s6_2022)
ewaso_data_2020 = format_water_level(ewaso_data_2020)
ewaso_data_2022 = format_water_level(ewaso_data_2022)

```

1. Filter the date range based on the water level data from first day of the water level data to the last day of the water level data
2. Choose stations within a certain radius of the gauging station 100 km for example get the resulting weather data
3. Get the stations with only 100 percent data no missing data
4. Remove the stations data with the value zero from beginning to end if the water level data has some values above zero
5. Calculate the correlation between the water level data and the weather data needs to be above 0 and have a lag of maximum 3 days
6. Plot the resulting figures 


### Choosing ewaso 2020 range
removing stations with missing data reduces from 1035 to 849 columns<br>
removing all zeros reduces from 849 to 604 columns<br>
columns with positive correlation reduces the number from 604 columns to 283 columns<br>
checking for lag reduces the columns to 80


```python
above, below = pipe.shed_stations(weather_stations_data,
                   muringato_data_s6_2022,
                   muringato_loc,
                   100,
                   lag=3
                   )

```


```python
below_stations = [i.split('_')[0] for i in below.keys()]
print(below_stations)
below_stations_metadata = fs.get_stations_info(multipleStations=below_stations)[['code', 'location.latitude', 'location.longitude']]
```

    ['TA00001', 'TA00023', 'TA00024', 'TA00025', 'TA00054', 'TA00056', 'TA00067', 'TA00077', 'TA00129', 'TA00147', 'TA00154', 'TA00155', 'TA00156', 'TA00166', 'TA00171', 'TA00189', 'TA00215', 'TA00222', 'TA00228', 'TA00230', 'TA00233', 'TA00250', 'TA00270', 'TA00270', 'TA00272', 'TA00272', 'TA00316', 'TA00317', 'TA00355', 'TA00459', 'TA00473', 'TA00480', 'TA00493', 'TA00494', 'TA00577', 'TA00601', 'TA00621', 'TA00653', 'TA00672', 'TA00676', 'TA00679', 'TA00692', 'TA00699', 'TA00704', 'TA00705', 'TA00711', 'TA00712', 'TA00712', 'TA00715', 'TA00717', 'TA00750', 'TA00751', 'TA00767']
    


```python
below_stations_metadata['distance']= below_stations_metadata.apply(lambda row: hs.haversine((muringato_loc[0], 
                                                                                             muringato_loc[1]), (row['location.latitude'], 
                                                                                                             row['location.longitude'])), axis=1)
below_stations_metadata.sort_values(by='distance')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>location.latitude</th>
      <th>location.longitude</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52</th>
      <td>TA00056</td>
      <td>-0.721656</td>
      <td>37.145585</td>
      <td>40.480889</td>
    </tr>
    <tr>
      <th>22</th>
      <td>TA00024</td>
      <td>-1.071731</td>
      <td>37.045578</td>
      <td>74.517013</td>
    </tr>
    <tr>
      <th>150</th>
      <td>TA00166</td>
      <td>-0.319508</td>
      <td>37.659139</td>
      <td>78.009238</td>
    </tr>
    <tr>
      <th>172</th>
      <td>TA00189</td>
      <td>-0.795260</td>
      <td>37.665930</td>
      <td>89.304790</td>
    </tr>
    <tr>
      <th>230</th>
      <td>TA00250</td>
      <td>-0.778940</td>
      <td>37.676738</td>
      <td>89.504935</td>
    </tr>
    <tr>
      <th>600</th>
      <td>TA00715</td>
      <td>-1.225618</td>
      <td>36.809065</td>
      <td>92.655456</td>
    </tr>
    <tr>
      <th>565</th>
      <td>TA00679</td>
      <td>-1.270835</td>
      <td>36.723916</td>
      <td>99.698089</td>
    </tr>
    <tr>
      <th>23</th>
      <td>TA00025</td>
      <td>-1.301839</td>
      <td>36.760200</td>
      <td>102.058383</td>
    </tr>
    <tr>
      <th>422</th>
      <td>TA00473</td>
      <td>-0.512371</td>
      <td>35.956813</td>
      <td>112.495996</td>
    </tr>
    <tr>
      <th>513</th>
      <td>TA00621</td>
      <td>-1.633020</td>
      <td>37.146185</td>
      <td>137.874253</td>
    </tr>
    <tr>
      <th>51</th>
      <td>TA00054</td>
      <td>-0.239342</td>
      <td>35.728897</td>
      <td>138.480985</td>
    </tr>
    <tr>
      <th>424</th>
      <td>TA00480</td>
      <td>-1.376152</td>
      <td>37.797646</td>
      <td>142.238019</td>
    </tr>
    <tr>
      <th>61</th>
      <td>TA00067</td>
      <td>-1.794285</td>
      <td>37.621211</td>
      <td>170.765765</td>
    </tr>
    <tr>
      <th>140</th>
      <td>TA00156</td>
      <td>-1.701123</td>
      <td>38.068339</td>
      <td>189.255406</td>
    </tr>
    <tr>
      <th>71</th>
      <td>TA00077</td>
      <td>-0.383066</td>
      <td>35.068406</td>
      <td>210.682047</td>
    </tr>
    <tr>
      <th>139</th>
      <td>TA00155</td>
      <td>-2.523037</td>
      <td>36.829437</td>
      <td>235.795373</td>
    </tr>
    <tr>
      <th>21</th>
      <td>TA00023</td>
      <td>-2.388550</td>
      <td>38.040767</td>
      <td>250.831198</td>
    </tr>
    <tr>
      <th>155</th>
      <td>TA00171</td>
      <td>-0.002710</td>
      <td>34.596908</td>
      <td>266.903936</td>
    </tr>
    <tr>
      <th>291</th>
      <td>TA00317</td>
      <td>0.040440</td>
      <td>34.371716</td>
      <td>292.394991</td>
    </tr>
    <tr>
      <th>0</th>
      <td>TA00001</td>
      <td>-1.123283</td>
      <td>34.397992</td>
      <td>296.112467</td>
    </tr>
    <tr>
      <th>652</th>
      <td>TA00767</td>
      <td>-2.671990</td>
      <td>38.369665</td>
      <td>296.467402</td>
    </tr>
    <tr>
      <th>290</th>
      <td>TA00316</td>
      <td>0.289862</td>
      <td>34.371222</td>
      <td>298.418648</td>
    </tr>
    <tr>
      <th>131</th>
      <td>TA00147</td>
      <td>0.449274</td>
      <td>34.282303</td>
      <td>312.905564</td>
    </tr>
    <tr>
      <th>117</th>
      <td>TA00129</td>
      <td>-3.390926</td>
      <td>37.717656</td>
      <td>342.264311</td>
    </tr>
    <tr>
      <th>138</th>
      <td>TA00154</td>
      <td>-4.231107</td>
      <td>37.847804</td>
      <td>436.466702</td>
    </tr>
    <tr>
      <th>211</th>
      <td>TA00230</td>
      <td>1.724690</td>
      <td>33.622000</td>
      <td>440.623881</td>
    </tr>
    <tr>
      <th>329</th>
      <td>TA00355</td>
      <td>3.498069</td>
      <td>35.843897</td>
      <td>451.651266</td>
    </tr>
    <tr>
      <th>544</th>
      <td>TA00653</td>
      <td>0.265062</td>
      <td>32.627203</td>
      <td>487.869319</td>
    </tr>
    <tr>
      <th>196</th>
      <td>TA00215</td>
      <td>0.052465</td>
      <td>32.440690</td>
      <td>505.441217</td>
    </tr>
    <tr>
      <th>203</th>
      <td>TA00222</td>
      <td>1.186240</td>
      <td>32.020330</td>
      <td>577.409865</td>
    </tr>
    <tr>
      <th>584</th>
      <td>TA00699</td>
      <td>-0.707570</td>
      <td>31.402138</td>
      <td>619.216128</td>
    </tr>
    <tr>
      <th>558</th>
      <td>TA00672</td>
      <td>-6.180302</td>
      <td>37.146832</td>
      <td>642.321296</td>
    </tr>
    <tr>
      <th>597</th>
      <td>TA00712</td>
      <td>-6.676308</td>
      <td>39.131552</td>
      <td>737.484276</td>
    </tr>
    <tr>
      <th>562</th>
      <td>TA00676</td>
      <td>-6.780374</td>
      <td>38.973512</td>
      <td>742.978650</td>
    </tr>
    <tr>
      <th>635</th>
      <td>TA00750</td>
      <td>-6.805316</td>
      <td>39.139843</td>
      <td>751.347364</td>
    </tr>
    <tr>
      <th>636</th>
      <td>TA00751</td>
      <td>-6.848668</td>
      <td>39.082174</td>
      <td>753.892793</td>
    </tr>
    <tr>
      <th>432</th>
      <td>TA00494</td>
      <td>-6.833860</td>
      <td>39.167475</td>
      <td>755.338586</td>
    </tr>
    <tr>
      <th>248</th>
      <td>TA00270</td>
      <td>-6.842390</td>
      <td>39.156760</td>
      <td>755.852180</td>
    </tr>
    <tr>
      <th>250</th>
      <td>TA00272</td>
      <td>-6.890039</td>
      <td>39.117927</td>
      <td>759.501414</td>
    </tr>
    <tr>
      <th>431</th>
      <td>TA00493</td>
      <td>-6.910845</td>
      <td>39.075597</td>
      <td>760.236606</td>
    </tr>
    <tr>
      <th>214</th>
      <td>TA00233</td>
      <td>3.453500</td>
      <td>31.251250</td>
      <td>766.277105</td>
    </tr>
    <tr>
      <th>209</th>
      <td>TA00228</td>
      <td>3.404720</td>
      <td>30.959600</td>
      <td>790.422401</td>
    </tr>
    <tr>
      <th>498</th>
      <td>TA00601</td>
      <td>-14.080148</td>
      <td>33.907593</td>
      <td>1557.147407</td>
    </tr>
    <tr>
      <th>602</th>
      <td>TA00717</td>
      <td>3.898305</td>
      <td>11.886437</td>
      <td>2827.236339</td>
    </tr>
    <tr>
      <th>590</th>
      <td>TA00705</td>
      <td>4.952251</td>
      <td>8.341692</td>
      <td>3234.191975</td>
    </tr>
    <tr>
      <th>481</th>
      <td>TA00577</td>
      <td>10.487147</td>
      <td>9.788223</td>
      <td>3240.086078</td>
    </tr>
    <tr>
      <th>589</th>
      <td>TA00704</td>
      <td>5.378602</td>
      <td>6.998292</td>
      <td>3388.907422</td>
    </tr>
    <tr>
      <th>596</th>
      <td>TA00711</td>
      <td>4.906530</td>
      <td>6.917064</td>
      <td>3389.011984</td>
    </tr>
    <tr>
      <th>410</th>
      <td>TA00459</td>
      <td>9.066148</td>
      <td>6.569080</td>
      <td>3526.820348</td>
    </tr>
    <tr>
      <th>577</th>
      <td>TA00692</td>
      <td>6.404114</td>
      <td>5.626307</td>
      <td>3559.025765</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Interactive visuals
import plotly.express as px
import plotly.graph_objects as go

fig = px.scatter_mapbox(below_stations_metadata, 
                        lat="location.latitude", 
                        lon="location.longitude", 
                        hover_name="code", 
                        hover_data=["distance"],
                        color_discrete_sequence=["fuchsia"],
                        zoom=8,
                        height=800,
                        )
# update marker size
fig.update_traces(marker=dict(size=10))
# add a point for the central station
fig.add_trace(go.Scattermapbox(
        lat=[muringato_loc[0]],
        lon=[muringato_loc[1]],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=14
        ),
        text=['Muringato gauging station'],
    ))

fig.update_layout(
    mapbox_style="carto-positron",
    margin={"r":0,"t":0,"l":0,"b":0},
    showlegend=False
)
fig.show()
```




```python
pipe.plot_figs(
    weather_stations_data,
    list(muringato_data_s6_2022['water_level']),
    list(below.keys()),
    date=dateutil.parser.parse(str(muringato_data_s6_2022.index[0])).strftime('%d-%m-%Y'), 
    save=False   
)
```

    Begin plotting!
    


    
![png](water_level_pipeline_files/water_level_pipeline_15_1.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_2.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_3.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_4.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_5.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_6.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_7.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_8.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_9.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_10.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_11.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_12.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_13.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_14.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_15.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_16.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_17.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_18.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_19.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_20.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_21.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_22.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_23.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_24.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_25.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_26.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_27.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_28.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_29.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_30.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_31.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_32.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_33.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_34.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_35.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_36.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_37.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_38.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_39.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_40.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_41.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_42.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_43.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_44.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_45.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_46.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_47.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_48.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_49.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_50.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_51.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_52.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_15_53.png)
    



```python
RADIUS = 100

ewaso_weather_data_2020 = weather_stations_data.loc[ewaso_data_2020.index[0]:ewaso_data_2020.index[-1]]
# ewaso stations within a particular radius
ewaso_tahmo_stations_2020 = pipe.stations_within_radius(RADIUS, ewaso_loc[0], ewaso_loc[1], df=False)
# Get stations without missing data
# ewaso weather data
ewaso_weather_data_2020_filtered = pipe.stations_data_check(stations_list=list(ewaso_tahmo_stations_2020), 
                                              percentage=1, data=ewaso_weather_data_2020
                                              )
# Check the sum of each column and drop columns with a sum of zero this is if the sum of water level is not equal to zero
ewaso_weather_data_2020_filtered = ewaso_weather_data_2020_filtered.loc[:, ewaso_weather_data_2020_filtered.sum() != 0]
```

    API request: services/assets/v2/stations
    


```python
import statsmodels.api as sm
def calculate_lag(weather_stations_data, water_level_data, lag=3, above=None, below=None):
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


```

Bringing all the functions together to create a pipeline


```python
def shed_stations(weather_stations_data, water_level_data,
                  gauging_station_coords, radius, lag=3,
                  percentage=1, above=None, below=None):
    # Filter the date range based on the water level data from first day of the water level data to the last day of the water level data
    weather_stations_data = weather_stations_data.loc[water_level_data.index[0]:water_level_data.index[-1]]
    # Filter the weather stations based on the radius
    lat, lon = gauging_station_coords[0], gauging_station_coords[1]
    weather_stations_data_list = pipe.stations_within_radius(radius, lat, lon, df=False)
    # get stations without missing data or the percentage of stations with missing data
    weather_stations_data_filtered = pipe.stations_data_check(stations_list=weather_stations_data_list,
                                                              percentage=percentage,
                                                              data=weather_stations_data)
    # Check the sum of each column and drop columns with a sum of zero this is if the sum of water level is not equal to zero
    weather_stations_data_filtered = weather_stations_data_filtered.loc[:, weather_stations_data_filtered.sum() != 0]

    # Filter the weather stations based on the lag and positive correlation
    above_threshold_lag, below_threshold_lag = calculate_lag(weather_stations_data_filtered, water_level_data, lag=lag)

    return above_threshold_lag, below_threshold_lag
```


```python
above_threshold_lag, below_threshold_lag = shed_stations(weather_stations_data, ewaso_data_2020, ewaso_loc, RADIUS, lag=3, percentage=1, above=True, below=False)
len(below_threshold_lag)
```

    API request: services/assets/v2/stations
    




    80



### Plot the figures


```python
pipe.plot_figs(
    weather_stations_data,
    list(ewaso_data_2020['water_level']),
    list(below_threshold_lag.keys()),
    date=dateutil.parser.parse(str(ewaso_data_2020.index[0])).strftime('%d-%m-%Y'), 
    save=True   
)
```

    Begin plotting!
    


    
![png](water_level_pipeline_files/water_level_pipeline_22_1.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_2.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_3.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_4.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_5.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_6.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_7.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_8.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_9.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_10.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_11.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_12.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_13.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_14.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_15.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_16.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_17.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_18.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_19.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_20.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_21.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_22.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_23.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_24.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_25.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_26.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_27.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_28.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_29.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_30.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_31.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_32.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_33.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_34.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_35.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_36.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_37.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_38.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_39.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_40.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_41.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_42.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_43.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_44.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_45.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_46.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_47.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_48.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_49.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_50.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_51.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_52.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_53.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_54.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_55.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_56.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_57.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_58.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_59.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_60.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_61.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_62.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_63.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_64.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_65.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_66.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_67.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_68.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_69.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_70.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_71.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_72.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_73.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_74.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_75.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_76.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_77.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_78.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_79.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_22_80.png)
    


Input water level data <br>
Input TAHMO station data <br>



```python
# plot the two with different colors
fig, ax = plt.subplots(figsize=(10, 10))
muringato_tahmo_stations.plot(kind='scatter',
                            x='location.longitude',
                            y='location.latitude',
                            color='blue',
                            alpha=0.7,
                            ax=ax)
ewaso_tahmo_stations.plot(kind='scatter',
                            x='location.longitude',
                            y='location.latitude',
                            color='red',
                            alpha=0.7,
                            ax=ax)
plt.show()
```


    
![png](water_level_pipeline_files/water_level_pipeline_24_0.png)
    


Apart from the completeness another method of validation by eliminating unusable sensors is checking for a positive correlation and lag
- The default lag is 3 days between a particular station and the gauging station
- The required format is a timeseries data 
- Provide the column names for evaluation format = [Date, data]
- with the change in parameters one can choose above or below threshold 


```python
def plot_figs(weather_stations, water_list, threshold_list, save=False, dpi=500, date='11-02-2021'):
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

```


```python
plot_figs(stations_df, lag_[list(lag_.keys())[0]]['water_list'], list(lag_.keys()), save=True, date='12-05-2020')
```

    Begin plotting!
    


    
![png](water_level_pipeline_files/water_level_pipeline_27_1.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_27_2.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_27_3.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_27_4.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_27_5.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_27_6.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_27_7.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_27_8.png)
    



    
![png](water_level_pipeline_files/water_level_pipeline_27_9.png)
    


Format to get the stations maetadata


```python
def filter_metadata(lag_keys):
    captured_list = [i.split('_')[0] for i in list(lag_keys)]
    return fs.get_stations_info(multipleStations=captured_list)
```


```python
filter_metadata(list(lag_.keys()))
```

    API request: services/assets/v2/stations
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>status</th>
      <th>installationdate</th>
      <th>elevationground</th>
      <th>sensorinstallations</th>
      <th>dataloggerinstallations</th>
      <th>creatorid</th>
      <th>created</th>
      <th>updaterid</th>
      <th>updated</th>
      <th>...</th>
      <th>location.countrycode</th>
      <th>location.zipcode</th>
      <th>location.latitude</th>
      <th>location.longitude</th>
      <th>location.elevationmsl</th>
      <th>location.note</th>
      <th>location.creatorid</th>
      <th>location.created</th>
      <th>location.updaterid</th>
      <th>location.updated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>TA00028</td>
      <td>1</td>
      <td>2015-08-31T00:00:00Z</td>
      <td>9.0</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>2018-12-11T08:35:17.888233Z</td>
      <td>2</td>
      <td>2018-12-11T08:35:17.888233Z</td>
      <td>...</td>
      <td>KE</td>
      <td></td>
      <td>0.055219</td>
      <td>37.136747</td>
      <td>2003.6</td>
      <td>{}</td>
      <td>2</td>
      <td>2018-10-26T13:32:16.15537Z</td>
      <td>37</td>
      <td>2022-06-30T11:11:50.27135Z</td>
    </tr>
    <tr>
      <th>27</th>
      <td>TA00029</td>
      <td>1</td>
      <td>2015-09-02T00:00:00Z</td>
      <td>2.0</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>2018-12-11T08:36:19.30342Z</td>
      <td>2</td>
      <td>2018-12-11T08:36:19.30342Z</td>
      <td>...</td>
      <td>KE</td>
      <td></td>
      <td>-0.500776</td>
      <td>36.587511</td>
      <td>2545.8</td>
      <td>{}</td>
      <td>2</td>
      <td>2018-10-26T13:33:31.451613Z</td>
      <td>37</td>
      <td>2022-02-28T12:25:09.578242Z</td>
    </tr>
    <tr>
      <th>53</th>
      <td>TA00057</td>
      <td>1</td>
      <td>2015-10-08T00:00:00Z</td>
      <td>2.0</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>2018-12-11T09:21:29.092833Z</td>
      <td>2</td>
      <td>2018-12-11T09:21:29.092833Z</td>
      <td>...</td>
      <td>KE</td>
      <td></td>
      <td>-1.253030</td>
      <td>36.856487</td>
      <td>1645.3</td>
      <td>{}</td>
      <td>2</td>
      <td>2018-10-29T09:13:33.768613Z</td>
      <td>2</td>
      <td>2022-07-26T07:34:06.603938Z</td>
    </tr>
    <tr>
      <th>68</th>
      <td>TA00074</td>
      <td>1</td>
      <td>2015-11-19T00:00:00Z</td>
      <td>2.0</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>2018-12-11T09:38:25.742397Z</td>
      <td>2</td>
      <td>2018-12-11T09:38:25.742397Z</td>
      <td>...</td>
      <td>KE</td>
      <td></td>
      <td>-0.566080</td>
      <td>37.074412</td>
      <td>1726.8</td>
      <td>{}</td>
      <td>2</td>
      <td>2018-10-29T10:35:28.49617Z</td>
      <td>2</td>
      <td>2022-07-26T07:38:42.100985Z</td>
    </tr>
    <tr>
      <th>74</th>
      <td>TA00080</td>
      <td>1</td>
      <td>2016-01-28T00:00:00Z</td>
      <td>2.0</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>2018-12-11T09:43:10.523398Z</td>
      <td>2</td>
      <td>2018-12-11T09:43:10.523398Z</td>
      <td>...</td>
      <td>KE</td>
      <td></td>
      <td>-1.087589</td>
      <td>36.818402</td>
      <td>1777.3</td>
      <td>{}</td>
      <td>2</td>
      <td>2018-10-29T10:53:47.845042Z</td>
      <td>37</td>
      <td>2022-02-28T13:07:04.709903Z</td>
    </tr>
    <tr>
      <th>150</th>
      <td>TA00166</td>
      <td>1</td>
      <td>2017-05-11T00:00:00Z</td>
      <td>2.0</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>2018-12-12T08:29:28.10697Z</td>
      <td>2</td>
      <td>2018-12-12T08:29:28.10697Z</td>
      <td>...</td>
      <td>KE</td>
      <td></td>
      <td>-0.319508</td>
      <td>37.659139</td>
      <td>1404.0</td>
      <td>{}</td>
      <td>2</td>
      <td>2018-11-10T08:47:37.949135Z</td>
      <td>2</td>
      <td>2018-11-10T08:47:37.949135Z</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 28 columns</p>
</div>




```python

```
