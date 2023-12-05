# Pedestrian Collision Prediction

Training a linear classifier with [GeoVex](https://openreview.net/forum?id=7bvWopYY1H) embeddings created using [srai](https://kraina-ai.github.io/srai/0.6.1/) python package.

###  This analysis is divided into the following parts:
- [Data Prep](Data-prep.ipynb) and creating synthetic data using [sdv](https://sdv.dev/SDV/)
- [Training GeoVex embeddings](Train-Geovex-Embeddings.ipynb) using [srai](https://github.com/kraina-ai/srai)
- [Training linear classifier](Train-Classifier.ipynb) to predict pedestrian collision hot spots based on urban features
- Visualization using [lonboard](https://developmentseed.org/lonboard/latest/)

###  Data sources
- [City of Calgary Open Data - Traffic Incidents](https://data.calgary.ca/Transportation-Transit/Traffic-Incidents/35ra-9556)
- [Climate Data](https://climatedata.ca/download/#station-download) 



![screenshot](collision-prediction-calgary.png)