<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<html><head><title>Python: module __init__</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
</head><body bgcolor="#f0f0f8">

<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="heading">
<tr bgcolor="#7799ee">
<td valign=bottom>&nbsp;<br>
<font color="#ffffff" face="helvetica, arial">&nbsp;<br><big><big><strong>__init__</strong></big></big></font></td
><td align=right valign=bottom
><font color="#ffffff" face="helvetica, arial"><a href=".">index</a><br><a href="file:c%3A%5Cusers%5Caustin%5Cdesktop%5Cagent%5Cpackages%5Cfilter_stations%5Cfilter_stations%5C__init__.py">c:\users\austin\desktop\agent\packages\filter_stations\filter_stations\__init__.py</a></font></td></tr></table>
    <p></p>
<p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#aa55cc">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#ffffff" face="helvetica, arial"><big><strong>Modules</strong></big></font></td></tr>
    
<tr><td bgcolor="#aa55cc"><tt>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</tt></td><td>&nbsp;</td>
<td width="100%"><table width="100%" summary="list"><tr><td width="25%" valign=top><a href="matplotlib.animation.html">matplotlib.animation</a><br>
<a href="argparse.html">argparse</a><br>
<a href="base64.html">base64</a><br>
<a href="datetime.html">datetime</a><br>
</td><td width="25%" valign=top><a href="dateutil.html">dateutil</a><br>
<a href="folium.html">folium</a><br>
<a href="gc.html">gc</a><br>
<a href="haversine.html">haversine</a><br>
</td><td width="25%" valign=top><a href="json.html">json</a><br>
<a href="math.html">math</a><br>
<a href="numpy.html">numpy</a><br>
<a href="os.html">os</a><br>
</td><td width="25%" valign=top><a href="pandas.html">pandas</a><br>
<a href="matplotlib.pyplot.html">matplotlib.pyplot</a><br>
<a href="requests.html">requests</a><br>
<a href="urllib.html">urllib</a><br>
</td></tr></table></td></tr></table><p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#ee77aa">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#ffffff" face="helvetica, arial"><big><strong>Classes</strong></big></font></td></tr>
    
<tr><td bgcolor="#ee77aa"><tt>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</tt></td><td>&nbsp;</td>
<td width="100%"><dl>
<dt><font face="helvetica, arial"><a href="builtins.html#object">builtins.object</a>
</font></dt><dd>
<dl>
<dt><font face="helvetica, arial"><a href="__init__.html#retreive_data">retreive_data</a>
</font></dt><dd>
<dl>
<dt><font face="helvetica, arial"><a href="__init__.html#Filter">Filter</a>
</font></dt><dt><font face="helvetica, arial"><a href="__init__.html#Interactive_maps">Interactive_maps</a>
</font></dt></dl>
</dd>
</dl>
</dd>
</dl>
 <p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#ffc8d8">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#000000" face="helvetica, arial"><a name="Filter">class <strong>Filter</strong></a>(<a href="__init__.html#retreive_data">retreive_data</a>)</font></td></tr>
    
<tr bgcolor="#ffc8d8"><td rowspan=2><tt>&nbsp;&nbsp;&nbsp;</tt></td>
<td colspan=2><tt><a href="#Filter">Filter</a>(apiKey,&nbsp;apiSecret)<br>
&nbsp;<br>
#&nbsp;Move&nbsp;the&nbsp;functions&nbsp;to&nbsp;a&nbsp;class<br>&nbsp;</tt></td></tr>
<tr><td>&nbsp;</td>
<td width="100%"><dl><dt>Method resolution order:</dt>
<dd><a href="__init__.html#Filter">Filter</a></dd>
<dd><a href="__init__.html#retreive_data">retreive_data</a></dd>
<dd><a href="builtins.html#object">builtins.object</a></dd>
</dl>
<hr>
Methods defined here:<br>
<dl><dt><a name="Filter-__init__"><strong>__init__</strong></a>(self, apiKey, apiSecret)</dt><dd><tt>Initialize&nbsp;self.&nbsp;&nbsp;See&nbsp;help(type(self))&nbsp;for&nbsp;accurate&nbsp;signature.</tt></dd></dl>

<dl><dt><a name="Filter-calculate_new_point"><strong>calculate_new_point</strong></a>(self, lat, lon, distance, bearing)</dt><dd><tt>Calculate&nbsp;a&nbsp;new&nbsp;point&nbsp;given&nbsp;a&nbsp;starting&nbsp;point,&nbsp;distance,&nbsp;and&nbsp;bearing.<br>
&nbsp;<br>
:param&nbsp;lat:&nbsp;starting&nbsp;latitude&nbsp;in&nbsp;degrees<br>
:param&nbsp;lon:&nbsp;starting&nbsp;longitude&nbsp;in&nbsp;degrees<br>
:param&nbsp;distance:&nbsp;distance&nbsp;to&nbsp;move&nbsp;in&nbsp;meters<br>
:param&nbsp;bearing:&nbsp;bearing&nbsp;to&nbsp;move&nbsp;in&nbsp;degrees&nbsp;(0&nbsp;is&nbsp;north)<br>
:return:&nbsp;tuple&nbsp;containing&nbsp;the&nbsp;new&nbsp;latitude&nbsp;and&nbsp;longitude&nbsp;in&nbsp;degrees</tt></dd></dl>

<dl><dt><a name="Filter-compute_filter"><strong>compute_filter</strong></a>(self, lat, lon, distance)</dt><dd><tt>Calculates&nbsp;the&nbsp;bounding&nbsp;box&nbsp;coordinates&nbsp;for&nbsp;a&nbsp;given&nbsp;location&nbsp;and&nbsp;distance.<br>
&nbsp;<br>
Parameters:<br>
lat&nbsp;(float):&nbsp;The&nbsp;latitude&nbsp;of&nbsp;the&nbsp;location.<br>
lon&nbsp;(float):&nbsp;The&nbsp;longitude&nbsp;of&nbsp;the&nbsp;location.<br>
distance&nbsp;(float):&nbsp;The&nbsp;distance&nbsp;from&nbsp;the&nbsp;location,&nbsp;in&nbsp;kilometers,&nbsp;to&nbsp;the&nbsp;edge&nbsp;of&nbsp;the&nbsp;bounding&nbsp;box.<br>
&nbsp;<br>
Returns:<br>
A&nbsp;tuple&nbsp;containing&nbsp;four&nbsp;floats&nbsp;representing&nbsp;the&nbsp;bounding&nbsp;box&nbsp;coordinates:&nbsp;(min_lat,&nbsp;min_lon,&nbsp;max_lat,&nbsp;max_lon).</tt></dd></dl>

<dl><dt><a name="Filter-filterStations"><strong>filterStations</strong></a>(self, address, distance, startDate=None, endDate=None, csvfile='KEcheck3.csv')</dt><dd><tt>This&nbsp;method&nbsp;filters&nbsp;weather&nbsp;station&nbsp;data&nbsp;within&nbsp;a&nbsp;certain&nbsp;distance&nbsp;from&nbsp;a&nbsp;given&nbsp;address.<br>
&nbsp;<br>
Parameters:<br>
address&nbsp;(str):&nbsp;Address&nbsp;to&nbsp;center&nbsp;the&nbsp;bounding&nbsp;box&nbsp;around.<br>
distance&nbsp;(float):&nbsp;The&nbsp;distance&nbsp;(in&nbsp;kilometers)&nbsp;from&nbsp;the&nbsp;center&nbsp;to&nbsp;the&nbsp;edge&nbsp;of&nbsp;the&nbsp;bounding&nbsp;box.<br>
startDate&nbsp;(str):&nbsp;The&nbsp;start&nbsp;date&nbsp;for&nbsp;filtering&nbsp;the&nbsp;weather&nbsp;station&nbsp;data&nbsp;in&nbsp;the&nbsp;format&nbsp;'YYYY-MM-DD'.<br>
endDate&nbsp;(str):&nbsp;The&nbsp;end&nbsp;date&nbsp;for&nbsp;filtering&nbsp;the&nbsp;weather&nbsp;station&nbsp;data&nbsp;in&nbsp;the&nbsp;format&nbsp;'YYYY-MM-DD'.<br>
csvfile&nbsp;(str):&nbsp;The&nbsp;name&nbsp;of&nbsp;the&nbsp;csv&nbsp;file&nbsp;containing&nbsp;the&nbsp;weather&nbsp;station&nbsp;data.<br>
&nbsp;<br>
Returns:<br>
pandas.DataFrame:&nbsp;The&nbsp;filtered&nbsp;weather&nbsp;station&nbsp;data&nbsp;within&nbsp;the&nbsp;bounding&nbsp;box.</tt></dd></dl>

<dl><dt><a name="Filter-filterStationsList"><strong>filterStationsList</strong></a>(self, address, distance=100)</dt><dd><tt>Filters&nbsp;stations&nbsp;based&nbsp;on&nbsp;their&nbsp;proximity&nbsp;to&nbsp;a&nbsp;given&nbsp;address&nbsp;and&nbsp;returns&nbsp;a&nbsp;list&nbsp;of&nbsp;station&nbsp;codes&nbsp;that&nbsp;fall&nbsp;within&nbsp;the&nbsp;specified&nbsp;distance.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;address&nbsp;(str):&nbsp;Address&nbsp;to&nbsp;filter&nbsp;stations&nbsp;by.<br>
&nbsp;&nbsp;&nbsp;&nbsp;distance&nbsp;(float,&nbsp;optional):&nbsp;Maximum&nbsp;distance&nbsp;(in&nbsp;kilometers)&nbsp;between&nbsp;the&nbsp;stations&nbsp;and&nbsp;the&nbsp;address.&nbsp;Default&nbsp;is&nbsp;100&nbsp;km.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;List&nbsp;of&nbsp;station&nbsp;codes&nbsp;that&nbsp;fall&nbsp;within&nbsp;the&nbsp;specified&nbsp;distance&nbsp;from&nbsp;the&nbsp;given&nbsp;address.</tt></dd></dl>

<dl><dt><a name="Filter-getLocation"><strong>getLocation</strong></a>(self, address)</dt><dd><tt>This&nbsp;method&nbsp;retrieves&nbsp;the&nbsp;latitude&nbsp;and&nbsp;longitude&nbsp;coordinates&nbsp;of&nbsp;a&nbsp;given&nbsp;address&nbsp;using&nbsp;the&nbsp;Nominatim&nbsp;API.<br>
&nbsp;<br>
Parameters:<br>
-----------<br>
address&nbsp;:&nbsp;str<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;address&nbsp;of&nbsp;the&nbsp;location&nbsp;you&nbsp;want&nbsp;to&nbsp;retrieve&nbsp;the&nbsp;coordinates&nbsp;for.<br>
&nbsp;&nbsp;&nbsp;&nbsp;<br>
Returns:<br>
--------<br>
Tuple&nbsp;(float,&nbsp;float)<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;latitude&nbsp;and&nbsp;longitude&nbsp;coordinates&nbsp;of&nbsp;the&nbsp;location.</tt></dd></dl>

<dl><dt><a name="Filter-get_stations_info"><strong>get_stations_info</strong></a>(self, station=None, multipleStations=[], countrycode=None)</dt><dd><tt>Retrieves&nbsp;information&nbsp;about&nbsp;weather&nbsp;stations&nbsp;from&nbsp;an&nbsp;API&nbsp;endpoint&nbsp;and&nbsp;returns&nbsp;relevant&nbsp;information&nbsp;based&nbsp;on&nbsp;the&nbsp;parameters&nbsp;passed&nbsp;to&nbsp;it.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;station&nbsp;(str,&nbsp;optional):&nbsp;Code&nbsp;for&nbsp;a&nbsp;single&nbsp;station&nbsp;to&nbsp;retrieve&nbsp;information&nbsp;for.&nbsp;Defaults&nbsp;to&nbsp;None.<br>
&nbsp;&nbsp;&nbsp;&nbsp;multipleStations&nbsp;(list,&nbsp;optional):&nbsp;List&nbsp;of&nbsp;station&nbsp;codes&nbsp;to&nbsp;retrieve&nbsp;information&nbsp;for&nbsp;multiple&nbsp;stations.&nbsp;Defaults&nbsp;to&nbsp;[].<br>
&nbsp;&nbsp;&nbsp;&nbsp;countrycode&nbsp;(str,&nbsp;optional):&nbsp;Country&nbsp;code&nbsp;to&nbsp;retrieve&nbsp;information&nbsp;for&nbsp;all&nbsp;stations&nbsp;located&nbsp;in&nbsp;the&nbsp;country.&nbsp;Defaults&nbsp;to&nbsp;None.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;pandas.DataFrame:&nbsp;DataFrame&nbsp;containing&nbsp;information&nbsp;about&nbsp;the&nbsp;requested&nbsp;weather&nbsp;stations.</tt></dd></dl>

<hr>
Methods inherited from <a href="__init__.html#retreive_data">retreive_data</a>:<br>
<dl><dt><a name="Filter-aggregate_variables"><strong>aggregate_variables</strong></a>(self, dataframe)</dt><dd><tt>Aggregates&nbsp;a&nbsp;pandas&nbsp;DataFrame&nbsp;of&nbsp;weather&nbsp;variables&nbsp;by&nbsp;summing&nbsp;values&nbsp;across&nbsp;each&nbsp;day.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;dataframe&nbsp;(pandas.DataFrame):&nbsp;DataFrame&nbsp;containing&nbsp;weather&nbsp;variable&nbsp;data.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;pandas.DataFrame:&nbsp;DataFrame&nbsp;containing&nbsp;aggregated&nbsp;weather&nbsp;variable&nbsp;data,&nbsp;summed&nbsp;by&nbsp;day.</tt></dd></dl>

<dl><dt><a name="Filter-get_measurements"><strong>get_measurements</strong></a>(self, station, startDate=None, endDate=None, variables=None, dataset='controlled', aggregate=False)</dt><dd><tt>Get&nbsp;measurements&nbsp;for&nbsp;a&nbsp;specified&nbsp;station&nbsp;and&nbsp;time&nbsp;period.<br>
&nbsp;<br>
:param&nbsp;station:&nbsp;The&nbsp;station&nbsp;ID&nbsp;for&nbsp;which&nbsp;to&nbsp;retrieve&nbsp;measurements.<br>
:type&nbsp;station:&nbsp;str<br>
:param&nbsp;startDate:&nbsp;The&nbsp;start&nbsp;date&nbsp;of&nbsp;the&nbsp;time&nbsp;period&nbsp;for&nbsp;which&nbsp;to&nbsp;retrieve&nbsp;measurements.<br>
:type&nbsp;startDate:&nbsp;datetime&nbsp;or&nbsp;str,&nbsp;optional<br>
:param&nbsp;endDate:&nbsp;The&nbsp;end&nbsp;date&nbsp;of&nbsp;the&nbsp;time&nbsp;period&nbsp;for&nbsp;which&nbsp;to&nbsp;retrieve&nbsp;measurements.<br>
:type&nbsp;endDate:&nbsp;datetime&nbsp;or&nbsp;str,&nbsp;optional<br>
:param&nbsp;variables:&nbsp;A&nbsp;list&nbsp;of&nbsp;variable&nbsp;shortcodes&nbsp;to&nbsp;retrieve&nbsp;measurements&nbsp;for.&nbsp;If&nbsp;None,&nbsp;all&nbsp;variables&nbsp;are&nbsp;retrieved.<br>
:type&nbsp;variables:&nbsp;list&nbsp;or&nbsp;None,&nbsp;optional<br>
:param&nbsp;dataset:&nbsp;The&nbsp;dataset&nbsp;to&nbsp;retrieve&nbsp;measurements&nbsp;from.&nbsp;Default&nbsp;is&nbsp;'controlled'.<br>
:type&nbsp;dataset:&nbsp;str,&nbsp;optional<br>
:param&nbsp;aggregate:&nbsp;Whether&nbsp;to&nbsp;aggregate&nbsp;variables&nbsp;by&nbsp;sensor&nbsp;ID.&nbsp;Default&nbsp;is&nbsp;False.<br>
:type&nbsp;aggregate:&nbsp;bool,&nbsp;optional<br>
:return:&nbsp;A&nbsp;Pandas&nbsp;DataFrame&nbsp;containing&nbsp;the&nbsp;requested&nbsp;measurements.<br>
:rtype:&nbsp;pandas.DataFrame</tt></dd></dl>

<dl><dt><a name="Filter-get_variables"><strong>get_variables</strong></a>(self)</dt><dd><tt>Retrieves&nbsp;information&nbsp;about&nbsp;available&nbsp;weather&nbsp;variables&nbsp;from&nbsp;an&nbsp;API&nbsp;endpoint.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;dict:&nbsp;Dictionary&nbsp;containing&nbsp;information&nbsp;about&nbsp;available&nbsp;weather&nbsp;variables,&nbsp;keyed&nbsp;by&nbsp;variable&nbsp;shortcode.</tt></dd></dl>

<dl><dt><a name="Filter-k_neighbours"><strong>k_neighbours</strong></a>(self, station, number=5)</dt><dd><tt>Returns&nbsp;a&nbsp;dictionary&nbsp;of&nbsp;the&nbsp;nearest&nbsp;neighbouring&nbsp;stations&nbsp;to&nbsp;the&nbsp;specified&nbsp;station.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;station&nbsp;(str):&nbsp;Code&nbsp;for&nbsp;the&nbsp;station&nbsp;to&nbsp;find&nbsp;neighbouring&nbsp;stations&nbsp;for.<br>
&nbsp;&nbsp;&nbsp;&nbsp;number&nbsp;(int,&nbsp;optional):&nbsp;Number&nbsp;of&nbsp;neighbouring&nbsp;stations&nbsp;to&nbsp;return.&nbsp;Defaults&nbsp;to&nbsp;5.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;dict:&nbsp;Dictionary&nbsp;containing&nbsp;the&nbsp;station&nbsp;codes&nbsp;and&nbsp;distances&nbsp;of&nbsp;the&nbsp;nearest&nbsp;neighbouring&nbsp;stations.</tt></dd></dl>

<dl><dt><a name="Filter-multiple_measurements"><strong>multiple_measurements</strong></a>(self, stations_list, csv_file, startDate, endDate, variables, dataset='controlled')</dt><dd><tt>Retrieves&nbsp;measurements&nbsp;for&nbsp;multiple&nbsp;stations&nbsp;and&nbsp;saves&nbsp;the&nbsp;aggregated&nbsp;data&nbsp;to&nbsp;a&nbsp;CSV&nbsp;file.<br>
&nbsp;<br>
Parameters:<br>
&nbsp;&nbsp;&nbsp;&nbsp;stations_list&nbsp;(list):&nbsp;A&nbsp;list&nbsp;of&nbsp;strings&nbsp;containing&nbsp;the&nbsp;names&nbsp;of&nbsp;the&nbsp;stations&nbsp;to&nbsp;retrieve&nbsp;data&nbsp;from.<br>
&nbsp;&nbsp;&nbsp;&nbsp;csv_file&nbsp;(str):&nbsp;The&nbsp;name&nbsp;of&nbsp;the&nbsp;CSV&nbsp;file&nbsp;to&nbsp;save&nbsp;the&nbsp;data&nbsp;to.<br>
&nbsp;&nbsp;&nbsp;&nbsp;startDate&nbsp;(str):&nbsp;The&nbsp;start&nbsp;date&nbsp;for&nbsp;the&nbsp;measurements,&nbsp;in&nbsp;the&nbsp;format&nbsp;'yyyy-mm-dd'.<br>
&nbsp;&nbsp;&nbsp;&nbsp;endDate&nbsp;(str):&nbsp;The&nbsp;end&nbsp;date&nbsp;for&nbsp;the&nbsp;measurements,&nbsp;in&nbsp;the&nbsp;format&nbsp;'yyyy-mm-dd'.<br>
&nbsp;&nbsp;&nbsp;&nbsp;variables&nbsp;(list):&nbsp;A&nbsp;list&nbsp;of&nbsp;strings&nbsp;containing&nbsp;the&nbsp;names&nbsp;of&nbsp;the&nbsp;variables&nbsp;to&nbsp;retrieve.<br>
&nbsp;&nbsp;&nbsp;&nbsp;dataset&nbsp;(str):&nbsp;The&nbsp;name&nbsp;of&nbsp;the&nbsp;dataset&nbsp;to&nbsp;retrieve&nbsp;the&nbsp;data&nbsp;from.&nbsp;Default&nbsp;is&nbsp;'controlled'.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;df&nbsp;(pandas.DataFrame):&nbsp;A&nbsp;DataFrame&nbsp;containing&nbsp;the&nbsp;aggregated&nbsp;data&nbsp;for&nbsp;all&nbsp;stations.<br>
&nbsp;<br>
Raises:<br>
&nbsp;&nbsp;&nbsp;&nbsp;ValueError:&nbsp;If&nbsp;stations_list&nbsp;is&nbsp;not&nbsp;a&nbsp;list.</tt></dd></dl>

<dl><dt><a name="Filter-raw_measurements"><strong>raw_measurements</strong></a>(self, station, startDate=None, endDate=None, variables=None)</dt></dl>

<hr>
Data descriptors inherited from <a href="__init__.html#retreive_data">retreive_data</a>:<br>
<dl><dt><strong>__dict__</strong></dt>
<dd><tt>dictionary&nbsp;for&nbsp;instance&nbsp;variables&nbsp;(if&nbsp;defined)</tt></dd>
</dl>
<dl><dt><strong>__weakref__</strong></dt>
<dd><tt>list&nbsp;of&nbsp;weak&nbsp;references&nbsp;to&nbsp;the&nbsp;object&nbsp;(if&nbsp;defined)</tt></dd>
</dl>
</td></tr></table> <p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#ffc8d8">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#000000" face="helvetica, arial"><a name="Interactive_maps">class <strong>Interactive_maps</strong></a>(<a href="__init__.html#retreive_data">retreive_data</a>)</font></td></tr>
    
<tr bgcolor="#ffc8d8"><td rowspan=2><tt>&nbsp;&nbsp;&nbsp;</tt></td>
<td colspan=2><tt><a href="#Interactive_maps">Interactive_maps</a>(apiKey,&nbsp;apiSecret)<br>
&nbsp;<br>
#&nbsp;A&nbsp;different&nbsp;class&nbsp;for&nbsp;visualisations<br>&nbsp;</tt></td></tr>
<tr><td>&nbsp;</td>
<td width="100%"><dl><dt>Method resolution order:</dt>
<dd><a href="__init__.html#Interactive_maps">Interactive_maps</a></dd>
<dd><a href="__init__.html#retreive_data">retreive_data</a></dd>
<dd><a href="builtins.html#object">builtins.object</a></dd>
</dl>
<hr>
Methods defined here:<br>
<dl><dt><a name="Interactive_maps-__init__"><strong>__init__</strong></a>(self, apiKey, apiSecret)</dt><dd><tt>Initialize&nbsp;self.&nbsp;&nbsp;See&nbsp;help(type(self))&nbsp;for&nbsp;accurate&nbsp;signature.</tt></dd></dl>

<dl><dt><a name="Interactive_maps-create_animation"><strong>create_animation</strong></a>(self, data, valid_sensors, day=100, T=10, interval=500)</dt><dd><tt>Creates&nbsp;an&nbsp;animation&nbsp;of&nbsp;pollutant&nbsp;levels&nbsp;for&nbsp;a&nbsp;given&nbsp;range&nbsp;of&nbsp;days&nbsp;and&nbsp;valid&nbsp;sensors.<br>
&nbsp;<br>
Parameters:<br>
data&nbsp;(DataFrame):&nbsp;A&nbsp;pandas&nbsp;DataFrame&nbsp;containing&nbsp;pollution&nbsp;data.<br>
valid_sensors&nbsp;(list):&nbsp;A&nbsp;list&nbsp;of&nbsp;valid&nbsp;sensor&nbsp;names.<br>
day&nbsp;(int):&nbsp;The&nbsp;starting&nbsp;day&nbsp;of&nbsp;the&nbsp;animation&nbsp;(default&nbsp;is&nbsp;100).<br>
T&nbsp;(int):&nbsp;The&nbsp;range&nbsp;of&nbsp;days&nbsp;for&nbsp;the&nbsp;animation&nbsp;(default&nbsp;is&nbsp;10).<br>
interval&nbsp;(int):&nbsp;The&nbsp;interval&nbsp;between&nbsp;frames&nbsp;in&nbsp;milliseconds&nbsp;(default&nbsp;is&nbsp;500).<br>
&nbsp;<br>
Returns:<br>
HTML:&nbsp;An&nbsp;HTML&nbsp;<a href="builtins.html#object">object</a>&nbsp;containing&nbsp;the&nbsp;animation.</tt></dd></dl>

<dl><dt><a name="Interactive_maps-draw_map"><strong>draw_map</strong></a>(self, map_center)</dt><dd><tt>Creates&nbsp;a&nbsp;Folium&nbsp;map&nbsp;centered&nbsp;on&nbsp;the&nbsp;specified&nbsp;location&nbsp;and&nbsp;adds&nbsp;markers&nbsp;for&nbsp;each&nbsp;weather&nbsp;station&nbsp;in&nbsp;the&nbsp;area.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;map_center:&nbsp;a&nbsp;tuple&nbsp;with&nbsp;the&nbsp;latitude&nbsp;and&nbsp;longitude&nbsp;of&nbsp;the&nbsp;center&nbsp;of&nbsp;the&nbsp;map<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;A&nbsp;Folium&nbsp;map&nbsp;<a href="builtins.html#object">object</a></tt></dd></dl>

<dl><dt><a name="Interactive_maps-encode_image"><strong>encode_image</strong></a>(self, ws, df_rainfall)</dt><dd><tt>Encodes&nbsp;a&nbsp;station's&nbsp;rainfall&nbsp;data&nbsp;plot&nbsp;as&nbsp;a&nbsp;base64-encoded&nbsp;image.<br>
&nbsp;<br>
Args:<br>
-&nbsp;ws&nbsp;(str):&nbsp;the&nbsp;code&nbsp;for&nbsp;the&nbsp;station&nbsp;to&nbsp;encode&nbsp;the&nbsp;image&nbsp;for<br>
-&nbsp;df_rainfall&nbsp;(pandas.DataFrame):&nbsp;a&nbsp;DataFrame&nbsp;containing&nbsp;the&nbsp;rainfall&nbsp;data&nbsp;for&nbsp;all&nbsp;stations<br>
&nbsp;<br>
Returns:<br>
-&nbsp;str:&nbsp;a&nbsp;string&nbsp;containing&nbsp;an&nbsp;HTML&nbsp;image&nbsp;tag&nbsp;with&nbsp;the&nbsp;encoded&nbsp;image&nbsp;data,&nbsp;or&nbsp;a&nbsp;message&nbsp;indicating&nbsp;no&nbsp;data&nbsp;is&nbsp;available&nbsp;for&nbsp;the&nbsp;given&nbsp;station</tt></dd></dl>

<dl><dt><a name="Interactive_maps-get_map"><strong>get_map</strong></a>(self, subset_list, start_date=None, end_date=None, data_values=False, csv_file='KEcheck3.csv', min_zoom=8, max_zoom=11, width=850, height=850, png_resolution=300)</dt><dd><tt>Creates&nbsp;a&nbsp;Folium&nbsp;map&nbsp;showing&nbsp;the&nbsp;locations&nbsp;of&nbsp;the&nbsp;weather&nbsp;stations&nbsp;in&nbsp;the&nbsp;given&nbsp;subsets.<br>
&nbsp;<br>
Parameters:<br>
-----------<br>
subset_list&nbsp;:&nbsp;list&nbsp;of&nbsp;lists&nbsp;of&nbsp;str<br>
&nbsp;&nbsp;&nbsp;&nbsp;List&nbsp;of&nbsp;subsets&nbsp;of&nbsp;weather&nbsp;stations,&nbsp;where&nbsp;each&nbsp;subset&nbsp;is&nbsp;a&nbsp;list&nbsp;of&nbsp;station&nbsp;codes.<br>
start_date&nbsp;:&nbsp;str,&nbsp;optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;Start&nbsp;date&nbsp;in&nbsp;the&nbsp;format&nbsp;YYYY-MM-DD,&nbsp;default&nbsp;is&nbsp;None.<br>
end_date&nbsp;:&nbsp;str,&nbsp;optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;End&nbsp;date&nbsp;in&nbsp;the&nbsp;format&nbsp;YYYY-MM-DD,&nbsp;default&nbsp;is&nbsp;None.<br>
data_values&nbsp;:&nbsp;bool,&nbsp;optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;If&nbsp;True,&nbsp;the&nbsp;map&nbsp;markers&nbsp;will&nbsp;display&nbsp;a&nbsp;plot&nbsp;of&nbsp;rainfall&nbsp;data,&nbsp;default&nbsp;is&nbsp;False.<br>
csv_file&nbsp;:&nbsp;str,&nbsp;optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;name&nbsp;of&nbsp;the&nbsp;CSV&nbsp;file&nbsp;containing&nbsp;the&nbsp;rainfall&nbsp;data,&nbsp;default&nbsp;is&nbsp;'KEcheck3.csv'.<br>
min_zoom&nbsp;:&nbsp;int,&nbsp;optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;minimum&nbsp;zoom&nbsp;level&nbsp;of&nbsp;the&nbsp;map,&nbsp;default&nbsp;is&nbsp;8.<br>
max_zoom&nbsp;:&nbsp;int,&nbsp;optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;maximum&nbsp;zoom&nbsp;level&nbsp;of&nbsp;the&nbsp;map,&nbsp;default&nbsp;is&nbsp;11.<br>
width&nbsp;:&nbsp;int,&nbsp;optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;width&nbsp;of&nbsp;the&nbsp;map&nbsp;in&nbsp;pixels,&nbsp;default&nbsp;is&nbsp;850.<br>
height&nbsp;:&nbsp;int,&nbsp;optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;height&nbsp;of&nbsp;the&nbsp;map&nbsp;in&nbsp;pixels,&nbsp;default&nbsp;is&nbsp;850.<br>
png_resolution&nbsp;:&nbsp;int,&nbsp;optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;resolution&nbsp;of&nbsp;the&nbsp;PNG&nbsp;image&nbsp;if&nbsp;data_values&nbsp;is&nbsp;True,&nbsp;default&nbsp;is&nbsp;300.<br>
&nbsp;<br>
Returns:<br>
--------<br>
my_map&nbsp;:&nbsp;folium.folium.Map<br>
&nbsp;&nbsp;&nbsp;&nbsp;A&nbsp;Folium&nbsp;map&nbsp;<a href="builtins.html#object">object</a>&nbsp;showing&nbsp;the&nbsp;locations&nbsp;of&nbsp;the&nbsp;weather&nbsp;stations&nbsp;in&nbsp;the&nbsp;given&nbsp;subsets.</tt></dd></dl>

<dl><dt><a name="Interactive_maps-plot_station"><strong>plot_station</strong></a>(self, ws, df_rainfall)</dt><dd><tt>Plot&nbsp;the&nbsp;rainfall&nbsp;data&nbsp;for&nbsp;a&nbsp;specific&nbsp;weather&nbsp;station.<br>
&nbsp;<br>
Args:<br>
-&nbsp;ws:&nbsp;string,&nbsp;the&nbsp;code&nbsp;of&nbsp;the&nbsp;weather&nbsp;station&nbsp;to&nbsp;plot<br>
-&nbsp;df_rainfall:&nbsp;DataFrame,&nbsp;a&nbsp;pandas&nbsp;DataFrame&nbsp;with&nbsp;rainfall&nbsp;data<br>
&nbsp;<br>
Returns:<br>
-&nbsp;None&nbsp;if&nbsp;no&nbsp;data&nbsp;is&nbsp;available&nbsp;for&nbsp;the&nbsp;specified&nbsp;station<br>
-&nbsp;a&nbsp;Matplotlib&nbsp;figure&nbsp;showing&nbsp;rainfall&nbsp;data&nbsp;for&nbsp;the&nbsp;specified&nbsp;station&nbsp;otherwise</tt></dd></dl>

<hr>
Methods inherited from <a href="__init__.html#retreive_data">retreive_data</a>:<br>
<dl><dt><a name="Interactive_maps-aggregate_variables"><strong>aggregate_variables</strong></a>(self, dataframe)</dt><dd><tt>Aggregates&nbsp;a&nbsp;pandas&nbsp;DataFrame&nbsp;of&nbsp;weather&nbsp;variables&nbsp;by&nbsp;summing&nbsp;values&nbsp;across&nbsp;each&nbsp;day.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;dataframe&nbsp;(pandas.DataFrame):&nbsp;DataFrame&nbsp;containing&nbsp;weather&nbsp;variable&nbsp;data.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;pandas.DataFrame:&nbsp;DataFrame&nbsp;containing&nbsp;aggregated&nbsp;weather&nbsp;variable&nbsp;data,&nbsp;summed&nbsp;by&nbsp;day.</tt></dd></dl>

<dl><dt><a name="Interactive_maps-get_measurements"><strong>get_measurements</strong></a>(self, station, startDate=None, endDate=None, variables=None, dataset='controlled', aggregate=False)</dt><dd><tt>Get&nbsp;measurements&nbsp;for&nbsp;a&nbsp;specified&nbsp;station&nbsp;and&nbsp;time&nbsp;period.<br>
&nbsp;<br>
:param&nbsp;station:&nbsp;The&nbsp;station&nbsp;ID&nbsp;for&nbsp;which&nbsp;to&nbsp;retrieve&nbsp;measurements.<br>
:type&nbsp;station:&nbsp;str<br>
:param&nbsp;startDate:&nbsp;The&nbsp;start&nbsp;date&nbsp;of&nbsp;the&nbsp;time&nbsp;period&nbsp;for&nbsp;which&nbsp;to&nbsp;retrieve&nbsp;measurements.<br>
:type&nbsp;startDate:&nbsp;datetime&nbsp;or&nbsp;str,&nbsp;optional<br>
:param&nbsp;endDate:&nbsp;The&nbsp;end&nbsp;date&nbsp;of&nbsp;the&nbsp;time&nbsp;period&nbsp;for&nbsp;which&nbsp;to&nbsp;retrieve&nbsp;measurements.<br>
:type&nbsp;endDate:&nbsp;datetime&nbsp;or&nbsp;str,&nbsp;optional<br>
:param&nbsp;variables:&nbsp;A&nbsp;list&nbsp;of&nbsp;variable&nbsp;shortcodes&nbsp;to&nbsp;retrieve&nbsp;measurements&nbsp;for.&nbsp;If&nbsp;None,&nbsp;all&nbsp;variables&nbsp;are&nbsp;retrieved.<br>
:type&nbsp;variables:&nbsp;list&nbsp;or&nbsp;None,&nbsp;optional<br>
:param&nbsp;dataset:&nbsp;The&nbsp;dataset&nbsp;to&nbsp;retrieve&nbsp;measurements&nbsp;from.&nbsp;Default&nbsp;is&nbsp;'controlled'.<br>
:type&nbsp;dataset:&nbsp;str,&nbsp;optional<br>
:param&nbsp;aggregate:&nbsp;Whether&nbsp;to&nbsp;aggregate&nbsp;variables&nbsp;by&nbsp;sensor&nbsp;ID.&nbsp;Default&nbsp;is&nbsp;False.<br>
:type&nbsp;aggregate:&nbsp;bool,&nbsp;optional<br>
:return:&nbsp;A&nbsp;Pandas&nbsp;DataFrame&nbsp;containing&nbsp;the&nbsp;requested&nbsp;measurements.<br>
:rtype:&nbsp;pandas.DataFrame</tt></dd></dl>

<dl><dt><a name="Interactive_maps-get_stations_info"><strong>get_stations_info</strong></a>(self, station=None, multipleStations=[], countrycode=None)</dt><dd><tt>Retrieves&nbsp;information&nbsp;about&nbsp;weather&nbsp;stations&nbsp;from&nbsp;an&nbsp;API&nbsp;endpoint&nbsp;and&nbsp;returns&nbsp;relevant&nbsp;information&nbsp;based&nbsp;on&nbsp;the&nbsp;parameters&nbsp;passed&nbsp;to&nbsp;it.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;station&nbsp;(str,&nbsp;optional):&nbsp;Code&nbsp;for&nbsp;a&nbsp;single&nbsp;station&nbsp;to&nbsp;retrieve&nbsp;information&nbsp;for.&nbsp;Defaults&nbsp;to&nbsp;None.<br>
&nbsp;&nbsp;&nbsp;&nbsp;multipleStations&nbsp;(list,&nbsp;optional):&nbsp;List&nbsp;of&nbsp;station&nbsp;codes&nbsp;to&nbsp;retrieve&nbsp;information&nbsp;for&nbsp;multiple&nbsp;stations.&nbsp;Defaults&nbsp;to&nbsp;[].<br>
&nbsp;&nbsp;&nbsp;&nbsp;countrycode&nbsp;(str,&nbsp;optional):&nbsp;Country&nbsp;code&nbsp;to&nbsp;retrieve&nbsp;information&nbsp;for&nbsp;all&nbsp;stations&nbsp;located&nbsp;in&nbsp;the&nbsp;country.&nbsp;Defaults&nbsp;to&nbsp;None.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;pandas.DataFrame:&nbsp;DataFrame&nbsp;containing&nbsp;information&nbsp;about&nbsp;the&nbsp;requested&nbsp;weather&nbsp;stations.</tt></dd></dl>

<dl><dt><a name="Interactive_maps-get_variables"><strong>get_variables</strong></a>(self)</dt><dd><tt>Retrieves&nbsp;information&nbsp;about&nbsp;available&nbsp;weather&nbsp;variables&nbsp;from&nbsp;an&nbsp;API&nbsp;endpoint.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;dict:&nbsp;Dictionary&nbsp;containing&nbsp;information&nbsp;about&nbsp;available&nbsp;weather&nbsp;variables,&nbsp;keyed&nbsp;by&nbsp;variable&nbsp;shortcode.</tt></dd></dl>

<dl><dt><a name="Interactive_maps-k_neighbours"><strong>k_neighbours</strong></a>(self, station, number=5)</dt><dd><tt>Returns&nbsp;a&nbsp;dictionary&nbsp;of&nbsp;the&nbsp;nearest&nbsp;neighbouring&nbsp;stations&nbsp;to&nbsp;the&nbsp;specified&nbsp;station.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;station&nbsp;(str):&nbsp;Code&nbsp;for&nbsp;the&nbsp;station&nbsp;to&nbsp;find&nbsp;neighbouring&nbsp;stations&nbsp;for.<br>
&nbsp;&nbsp;&nbsp;&nbsp;number&nbsp;(int,&nbsp;optional):&nbsp;Number&nbsp;of&nbsp;neighbouring&nbsp;stations&nbsp;to&nbsp;return.&nbsp;Defaults&nbsp;to&nbsp;5.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;dict:&nbsp;Dictionary&nbsp;containing&nbsp;the&nbsp;station&nbsp;codes&nbsp;and&nbsp;distances&nbsp;of&nbsp;the&nbsp;nearest&nbsp;neighbouring&nbsp;stations.</tt></dd></dl>

<dl><dt><a name="Interactive_maps-multiple_measurements"><strong>multiple_measurements</strong></a>(self, stations_list, csv_file, startDate, endDate, variables, dataset='controlled')</dt><dd><tt>Retrieves&nbsp;measurements&nbsp;for&nbsp;multiple&nbsp;stations&nbsp;and&nbsp;saves&nbsp;the&nbsp;aggregated&nbsp;data&nbsp;to&nbsp;a&nbsp;CSV&nbsp;file.<br>
&nbsp;<br>
Parameters:<br>
&nbsp;&nbsp;&nbsp;&nbsp;stations_list&nbsp;(list):&nbsp;A&nbsp;list&nbsp;of&nbsp;strings&nbsp;containing&nbsp;the&nbsp;names&nbsp;of&nbsp;the&nbsp;stations&nbsp;to&nbsp;retrieve&nbsp;data&nbsp;from.<br>
&nbsp;&nbsp;&nbsp;&nbsp;csv_file&nbsp;(str):&nbsp;The&nbsp;name&nbsp;of&nbsp;the&nbsp;CSV&nbsp;file&nbsp;to&nbsp;save&nbsp;the&nbsp;data&nbsp;to.<br>
&nbsp;&nbsp;&nbsp;&nbsp;startDate&nbsp;(str):&nbsp;The&nbsp;start&nbsp;date&nbsp;for&nbsp;the&nbsp;measurements,&nbsp;in&nbsp;the&nbsp;format&nbsp;'yyyy-mm-dd'.<br>
&nbsp;&nbsp;&nbsp;&nbsp;endDate&nbsp;(str):&nbsp;The&nbsp;end&nbsp;date&nbsp;for&nbsp;the&nbsp;measurements,&nbsp;in&nbsp;the&nbsp;format&nbsp;'yyyy-mm-dd'.<br>
&nbsp;&nbsp;&nbsp;&nbsp;variables&nbsp;(list):&nbsp;A&nbsp;list&nbsp;of&nbsp;strings&nbsp;containing&nbsp;the&nbsp;names&nbsp;of&nbsp;the&nbsp;variables&nbsp;to&nbsp;retrieve.<br>
&nbsp;&nbsp;&nbsp;&nbsp;dataset&nbsp;(str):&nbsp;The&nbsp;name&nbsp;of&nbsp;the&nbsp;dataset&nbsp;to&nbsp;retrieve&nbsp;the&nbsp;data&nbsp;from.&nbsp;Default&nbsp;is&nbsp;'controlled'.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;df&nbsp;(pandas.DataFrame):&nbsp;A&nbsp;DataFrame&nbsp;containing&nbsp;the&nbsp;aggregated&nbsp;data&nbsp;for&nbsp;all&nbsp;stations.<br>
&nbsp;<br>
Raises:<br>
&nbsp;&nbsp;&nbsp;&nbsp;ValueError:&nbsp;If&nbsp;stations_list&nbsp;is&nbsp;not&nbsp;a&nbsp;list.</tt></dd></dl>

<dl><dt><a name="Interactive_maps-raw_measurements"><strong>raw_measurements</strong></a>(self, station, startDate=None, endDate=None, variables=None)</dt></dl>

<hr>
Data descriptors inherited from <a href="__init__.html#retreive_data">retreive_data</a>:<br>
<dl><dt><strong>__dict__</strong></dt>
<dd><tt>dictionary&nbsp;for&nbsp;instance&nbsp;variables&nbsp;(if&nbsp;defined)</tt></dd>
</dl>
<dl><dt><strong>__weakref__</strong></dt>
<dd><tt>list&nbsp;of&nbsp;weak&nbsp;references&nbsp;to&nbsp;the&nbsp;object&nbsp;(if&nbsp;defined)</tt></dd>
</dl>
</td></tr></table> <p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#ffc8d8">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#000000" face="helvetica, arial"><a name="retreive_data">class <strong>retreive_data</strong></a>(<a href="builtins.html#object">builtins.object</a>)</font></td></tr>
    
<tr bgcolor="#ffc8d8"><td rowspan=2><tt>&nbsp;&nbsp;&nbsp;</tt></td>
<td colspan=2><tt><a href="#retreive_data">retreive_data</a>(apiKey,&nbsp;apiSecret)<br>
&nbsp;<br>
#&nbsp;Get&nbsp;data&nbsp;class<br>&nbsp;</tt></td></tr>
<tr><td>&nbsp;</td>
<td width="100%">Methods defined here:<br>
<dl><dt><a name="retreive_data-__init__"><strong>__init__</strong></a>(self, apiKey, apiSecret)</dt><dd><tt>Initialize&nbsp;self.&nbsp;&nbsp;See&nbsp;help(type(self))&nbsp;for&nbsp;accurate&nbsp;signature.</tt></dd></dl>

<dl><dt><a name="retreive_data-aggregate_variables"><strong>aggregate_variables</strong></a>(self, dataframe)</dt><dd><tt>Aggregates&nbsp;a&nbsp;pandas&nbsp;DataFrame&nbsp;of&nbsp;weather&nbsp;variables&nbsp;by&nbsp;summing&nbsp;values&nbsp;across&nbsp;each&nbsp;day.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;dataframe&nbsp;(pandas.DataFrame):&nbsp;DataFrame&nbsp;containing&nbsp;weather&nbsp;variable&nbsp;data.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;pandas.DataFrame:&nbsp;DataFrame&nbsp;containing&nbsp;aggregated&nbsp;weather&nbsp;variable&nbsp;data,&nbsp;summed&nbsp;by&nbsp;day.</tt></dd></dl>

<dl><dt><a name="retreive_data-get_measurements"><strong>get_measurements</strong></a>(self, station, startDate=None, endDate=None, variables=None, dataset='controlled', aggregate=False)</dt><dd><tt>Get&nbsp;measurements&nbsp;for&nbsp;a&nbsp;specified&nbsp;station&nbsp;and&nbsp;time&nbsp;period.<br>
&nbsp;<br>
:param&nbsp;station:&nbsp;The&nbsp;station&nbsp;ID&nbsp;for&nbsp;which&nbsp;to&nbsp;retrieve&nbsp;measurements.<br>
:type&nbsp;station:&nbsp;str<br>
:param&nbsp;startDate:&nbsp;The&nbsp;start&nbsp;date&nbsp;of&nbsp;the&nbsp;time&nbsp;period&nbsp;for&nbsp;which&nbsp;to&nbsp;retrieve&nbsp;measurements.<br>
:type&nbsp;startDate:&nbsp;datetime&nbsp;or&nbsp;str,&nbsp;optional<br>
:param&nbsp;endDate:&nbsp;The&nbsp;end&nbsp;date&nbsp;of&nbsp;the&nbsp;time&nbsp;period&nbsp;for&nbsp;which&nbsp;to&nbsp;retrieve&nbsp;measurements.<br>
:type&nbsp;endDate:&nbsp;datetime&nbsp;or&nbsp;str,&nbsp;optional<br>
:param&nbsp;variables:&nbsp;A&nbsp;list&nbsp;of&nbsp;variable&nbsp;shortcodes&nbsp;to&nbsp;retrieve&nbsp;measurements&nbsp;for.&nbsp;If&nbsp;None,&nbsp;all&nbsp;variables&nbsp;are&nbsp;retrieved.<br>
:type&nbsp;variables:&nbsp;list&nbsp;or&nbsp;None,&nbsp;optional<br>
:param&nbsp;dataset:&nbsp;The&nbsp;dataset&nbsp;to&nbsp;retrieve&nbsp;measurements&nbsp;from.&nbsp;Default&nbsp;is&nbsp;'controlled'.<br>
:type&nbsp;dataset:&nbsp;str,&nbsp;optional<br>
:param&nbsp;aggregate:&nbsp;Whether&nbsp;to&nbsp;aggregate&nbsp;variables&nbsp;by&nbsp;sensor&nbsp;ID.&nbsp;Default&nbsp;is&nbsp;False.<br>
:type&nbsp;aggregate:&nbsp;bool,&nbsp;optional<br>
:return:&nbsp;A&nbsp;Pandas&nbsp;DataFrame&nbsp;containing&nbsp;the&nbsp;requested&nbsp;measurements.<br>
:rtype:&nbsp;pandas.DataFrame</tt></dd></dl>

<dl><dt><a name="retreive_data-get_stations_info"><strong>get_stations_info</strong></a>(self, station=None, multipleStations=[], countrycode=None)</dt><dd><tt>Retrieves&nbsp;information&nbsp;about&nbsp;weather&nbsp;stations&nbsp;from&nbsp;an&nbsp;API&nbsp;endpoint&nbsp;and&nbsp;returns&nbsp;relevant&nbsp;information&nbsp;based&nbsp;on&nbsp;the&nbsp;parameters&nbsp;passed&nbsp;to&nbsp;it.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;station&nbsp;(str,&nbsp;optional):&nbsp;Code&nbsp;for&nbsp;a&nbsp;single&nbsp;station&nbsp;to&nbsp;retrieve&nbsp;information&nbsp;for.&nbsp;Defaults&nbsp;to&nbsp;None.<br>
&nbsp;&nbsp;&nbsp;&nbsp;multipleStations&nbsp;(list,&nbsp;optional):&nbsp;List&nbsp;of&nbsp;station&nbsp;codes&nbsp;to&nbsp;retrieve&nbsp;information&nbsp;for&nbsp;multiple&nbsp;stations.&nbsp;Defaults&nbsp;to&nbsp;[].<br>
&nbsp;&nbsp;&nbsp;&nbsp;countrycode&nbsp;(str,&nbsp;optional):&nbsp;Country&nbsp;code&nbsp;to&nbsp;retrieve&nbsp;information&nbsp;for&nbsp;all&nbsp;stations&nbsp;located&nbsp;in&nbsp;the&nbsp;country.&nbsp;Defaults&nbsp;to&nbsp;None.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;pandas.DataFrame:&nbsp;DataFrame&nbsp;containing&nbsp;information&nbsp;about&nbsp;the&nbsp;requested&nbsp;weather&nbsp;stations.</tt></dd></dl>

<dl><dt><a name="retreive_data-get_variables"><strong>get_variables</strong></a>(self)</dt><dd><tt>Retrieves&nbsp;information&nbsp;about&nbsp;available&nbsp;weather&nbsp;variables&nbsp;from&nbsp;an&nbsp;API&nbsp;endpoint.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;dict:&nbsp;Dictionary&nbsp;containing&nbsp;information&nbsp;about&nbsp;available&nbsp;weather&nbsp;variables,&nbsp;keyed&nbsp;by&nbsp;variable&nbsp;shortcode.</tt></dd></dl>

<dl><dt><a name="retreive_data-k_neighbours"><strong>k_neighbours</strong></a>(self, station, number=5)</dt><dd><tt>Returns&nbsp;a&nbsp;dictionary&nbsp;of&nbsp;the&nbsp;nearest&nbsp;neighbouring&nbsp;stations&nbsp;to&nbsp;the&nbsp;specified&nbsp;station.<br>
&nbsp;<br>
Args:<br>
&nbsp;&nbsp;&nbsp;&nbsp;station&nbsp;(str):&nbsp;Code&nbsp;for&nbsp;the&nbsp;station&nbsp;to&nbsp;find&nbsp;neighbouring&nbsp;stations&nbsp;for.<br>
&nbsp;&nbsp;&nbsp;&nbsp;number&nbsp;(int,&nbsp;optional):&nbsp;Number&nbsp;of&nbsp;neighbouring&nbsp;stations&nbsp;to&nbsp;return.&nbsp;Defaults&nbsp;to&nbsp;5.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;dict:&nbsp;Dictionary&nbsp;containing&nbsp;the&nbsp;station&nbsp;codes&nbsp;and&nbsp;distances&nbsp;of&nbsp;the&nbsp;nearest&nbsp;neighbouring&nbsp;stations.</tt></dd></dl>

<dl><dt><a name="retreive_data-multiple_measurements"><strong>multiple_measurements</strong></a>(self, stations_list, csv_file, startDate, endDate, variables, dataset='controlled')</dt><dd><tt>Retrieves&nbsp;measurements&nbsp;for&nbsp;multiple&nbsp;stations&nbsp;and&nbsp;saves&nbsp;the&nbsp;aggregated&nbsp;data&nbsp;to&nbsp;a&nbsp;CSV&nbsp;file.<br>
&nbsp;<br>
Parameters:<br>
&nbsp;&nbsp;&nbsp;&nbsp;stations_list&nbsp;(list):&nbsp;A&nbsp;list&nbsp;of&nbsp;strings&nbsp;containing&nbsp;the&nbsp;names&nbsp;of&nbsp;the&nbsp;stations&nbsp;to&nbsp;retrieve&nbsp;data&nbsp;from.<br>
&nbsp;&nbsp;&nbsp;&nbsp;csv_file&nbsp;(str):&nbsp;The&nbsp;name&nbsp;of&nbsp;the&nbsp;CSV&nbsp;file&nbsp;to&nbsp;save&nbsp;the&nbsp;data&nbsp;to.<br>
&nbsp;&nbsp;&nbsp;&nbsp;startDate&nbsp;(str):&nbsp;The&nbsp;start&nbsp;date&nbsp;for&nbsp;the&nbsp;measurements,&nbsp;in&nbsp;the&nbsp;format&nbsp;'yyyy-mm-dd'.<br>
&nbsp;&nbsp;&nbsp;&nbsp;endDate&nbsp;(str):&nbsp;The&nbsp;end&nbsp;date&nbsp;for&nbsp;the&nbsp;measurements,&nbsp;in&nbsp;the&nbsp;format&nbsp;'yyyy-mm-dd'.<br>
&nbsp;&nbsp;&nbsp;&nbsp;variables&nbsp;(list):&nbsp;A&nbsp;list&nbsp;of&nbsp;strings&nbsp;containing&nbsp;the&nbsp;names&nbsp;of&nbsp;the&nbsp;variables&nbsp;to&nbsp;retrieve.<br>
&nbsp;&nbsp;&nbsp;&nbsp;dataset&nbsp;(str):&nbsp;The&nbsp;name&nbsp;of&nbsp;the&nbsp;dataset&nbsp;to&nbsp;retrieve&nbsp;the&nbsp;data&nbsp;from.&nbsp;Default&nbsp;is&nbsp;'controlled'.<br>
&nbsp;<br>
Returns:<br>
&nbsp;&nbsp;&nbsp;&nbsp;df&nbsp;(pandas.DataFrame):&nbsp;A&nbsp;DataFrame&nbsp;containing&nbsp;the&nbsp;aggregated&nbsp;data&nbsp;for&nbsp;all&nbsp;stations.<br>
&nbsp;<br>
Raises:<br>
&nbsp;&nbsp;&nbsp;&nbsp;ValueError:&nbsp;If&nbsp;stations_list&nbsp;is&nbsp;not&nbsp;a&nbsp;list.</tt></dd></dl>

<dl><dt><a name="retreive_data-raw_measurements"><strong>raw_measurements</strong></a>(self, station, startDate=None, endDate=None, variables=None)</dt></dl>

<hr>
Data descriptors defined here:<br>
<dl><dt><strong>__dict__</strong></dt>
<dd><tt>dictionary&nbsp;for&nbsp;instance&nbsp;variables&nbsp;(if&nbsp;defined)</tt></dd>
</dl>
<dl><dt><strong>__weakref__</strong></dt>
<dd><tt>list&nbsp;of&nbsp;weak&nbsp;references&nbsp;to&nbsp;the&nbsp;object&nbsp;(if&nbsp;defined)</tt></dd>
</dl>
</td></tr></table></td></tr></table><p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#eeaa77">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#ffffff" face="helvetica, arial"><big><strong>Functions</strong></big></font></td></tr>
    
<tr><td bgcolor="#eeaa77"><tt>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</tt></td><td>&nbsp;</td>
<td width="100%"><dl><dt><a name="-parse_args"><strong>parse_args</strong></a>()</dt></dl>
</td></tr></table><p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#55aa55">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#ffffff" face="helvetica, arial"><big><strong>Data</strong></big></font></td></tr>
    
<tr><td bgcolor="#55aa55"><tt>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</tt></td><td>&nbsp;</td>
<td width="100%"><strong>API_BASE_URL</strong> = 'https://datahub.tahmo.org'<br>
<strong>API_MAX_PERIOD</strong> = '365D'<br>
<strong>endpoints</strong> = {'DATA_COMPLETE': 'custom/sensordx/latestmeasurements', 'STATION_INFO': 'services/assets/v2/stations', 'STATION_STATUS': 'custom/stations/status', 'VARIABLES': 'services/assets/v2/variables', 'WEATHER_DATA': 'services/measurements/v2/stations'}</td></tr></table>
</body></html>
