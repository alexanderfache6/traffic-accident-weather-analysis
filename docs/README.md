Notice: This project is still in progress.

# Overview


# Datasets

[US Accidents](https://www.kaggle.com/sobhanmoosavi/us-accidents)

Features

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| ID | Source | TMC | Severity | Start_Time | End_Time | Start_Lat | Stop_Lng | End_Lat | End_Lng |
| 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
| Distance(mi) | Description | Number | Street | Side | City | County | State | Zipcode | Country |
| 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
| Timezone | Airport_Code | Weather_Timestamp | Temperature(F) | Wind_Chill(F) | Humidity(%) | Pressure(in) | Visibility(mi) | Wind_Direction | Wind_Speed(mph) |
| 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
| Precipitation(in) | Weather_Condition | Amenity | Bumpy | Crossing | Give_Way | Junction | No_Exit | Railway | Roundabout |
| 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
| Station | Stop | Traffic_Calming | Traffic_Signal | Turning_Loop | Sunrise_Sunset | Civil_Twilight | Nautical_Twilight | Astronomical_Twilight |

[US Weather Events](https://www.kaggle.com/sobhanmoosavi/us-weather-events)

# Modules

## 1 Feature Extraction, Dimensionality Reduction, Feature Ranking

- Preprocessing
  - fill in missing categorical data with mode
  - clean up date time objects
  - replace True/False data with 1/0
  - one-hot encoding for categorical data
- PCA
- Random Forest

## 2 Supervised Learning

### KNN

### Neural Network

## X Visualization

### US
![alt text](https://github.com/alexanderfache6/traffic-accident-weather-analysis/blob/master/docs/map_usa.png "Map of US Accidents")
- Red: Severity 4
- Yellow: Severity 3
- Green: Severity: 2
- Blue: Severity: 1

### Houston
![alt text](https://github.com/alexanderfache6/traffic-accident-weather-analysis/blob/master/docs/map_true.png "Map of Houston Accidents")

### DBSCAN of Latitude and Longitude - Houston

![alt text](https://github.com/alexanderfache6/traffic-accident-weather-analysis/blob/master/docs/map_dbscan_e001_s10.png "DBSCAN of Houston Accidents")

# References

https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d
