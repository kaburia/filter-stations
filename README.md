## Documentation
You can find the documentation for the project by following this link<br>
https://filter-stations.readthedocs.io/en/latest/

Getting Started
---------------
All methods require an API key and secret, which can be obtained by contacting TAHMO. <br>
- The ```RetrieveData``` class is used to retrieve data from the TAHMO API endpoints.<br> 
- The ```RainLoader``` class is used to get our DSAIL unified weather dataset from HuggingFace (See the documentation for more information on this) <br>
- The ```Kieni``` class is used to get weather data for stations 100km around Kieni from the central point.with water level data.<br>
<!-- - The ```Interactive_maps``` class is used to plot weather stations on an interactive map.<br>
- The ```Water_level``` class is used to retrieve water level data and coordinates of gauging stations.<br> -->

<!-- For instructions on shedding weather stations based on your water level data and gauging station coordinates, please refer to the [water_level_pipeline.md](https://github.com/kaburia/filter-stations/blob/main/water_level_pipeline.md) file. -->
<br>

To get started on the module test it out on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KUUtvozBTePyezc1i5hhDuFFSWLzEcXH?usp=sharing)


For earlier versions (<= v0.6.2) use the link below for documentation <br>

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
