 Note: This project is still in progress.

# Motivation
 In a 2019 Business Insider report, the city of Atlanta is reported to be ranked as number 11 on the most traffic congested cities in the United States. Unfortunately, thousands of Atlanta commuters are plagued with motor vehicle collisions every year in various traffic conditions. According to a published 2016 traffic report of Fulton County, Atlanta has faced an estimate of 60,984 automotive accidents with 12,875 injuries. One explanation for higher crash rates in Atlanta roads is that extreme road conditions due to weather (e.g. rain, snow, ice) create potential safety hazards. Such potential safety hazards include, but not limited to: driver(s) lose complete control of vehicle(s), improper lane change, or obstruction of visibility. The United States Department of Transportation Road Weather Management Program reports that annual averages from 2007-2016 show 15% of vehicle crashes occurred due to wet pavements with 10% due to rain, 4% due to snow, and 3% due to ice (“How Do Weather Events Impact Roads?”, 2018).

### What are you trying to do to tackle with your project motivation or problem?
By implementing a real-time updating machine learning model fed with informative data, users (drivers) can explore the most dangerous locations along their commutes during extreme weather conditions to either avoid or take extra precautions. As more and more Atlanta drivers become aware of road conditions along their respective routes, there could be a significant reduction in the number of automotive accidents, injuries, and fatalities. Our team has developed a risk assessment for regions that drivers can use to evaluate driving conditions and take necessary precautions.

### What have people already done?
We have used relevant features such as precipitation, weather condition, time of day, road type, severity and so forth to calculate risk scores along Atlanta roads. WILL COME BACK TO THIS AFTER GROUP DISCUSSION MEETING

# Datasets

### [US Accidents](https://www.kaggle.com/sobhanmoosavi/us-accidents)

3.0 million records of traffic accidents across 49 US states from February 2016 to December 2019.

#### Features

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

# Approach

### Why do you think your approach can effectively solve your problem?
### What is new in your approach?

# Modules

## 1 Feature Extraction, Dimensionality Reduction, Feature Ranking

### Preprocessing
  - fill in missing categorical data with mode of category
  - clean up date time objects
    - split into year, month, day, hour, minute, second
  - replace True/False data with 1/0
  - one-hot encoding for categorical data
    - ex. Sunrise_Sunset = {Day, Night}. Turn into Sunrise_Sunset_Day = True/False, Sunrise_Sunset_Night = True/False

### Principle Component Analysis (PCA)

- Aims to select principal components in Z space to attain the largest possible variance.

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/module1_data/GA/correlation_original.png?token=AGCBXXQBKWDUN7AHONNL5IK6TNUPK "Original Correlation")

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/module1_data/GA/correlation_pca.png?token=AGCBXXTXJNVRX6J2XDXBAUK6TNUQK "PCA Correlation")

### Random Forest Forward Feature Selection

- Rank features based on importance.
- Importance is determined based on value of feature in constructing decision tree.

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/module1_data/GA/random_forest_f102.png?token=AGCBXXRIL3BKBTJ4T63UACC6TNURM "Random Forest Feature Importance")

## 2 Supervised Learning

### Logistic Regression



#### Preprocessing



#### Implementation



#### Results



### Support Vector Machine

#### Preprocessing



#### Implementation



#### Results



### Decision Trees

#### Preprocessing



#### Implementation



#### Results



## X Visualization

### US

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/map_usa.png?token=AGCBXXR5JQCTMUPC43ZJ3AS6TNULW "Map of US Accidents")

- Red: Severity 4
- Yellow: Severity 3
- Green: Severity: 2
- Blue: Severity: 1

### Georgia

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/GA.png?token=AGCBXXSWMABYG3ZU767G5OS6TNUCY "Map of GA Accidents")

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/colormap.png?token=AGCBXXWRWQFSQK6CDVVVR5S6TNUNM "Severity Color")

- 1 | 2 | 3 | 4

# Conclusion



# References

- [1] Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, arXiv preprint arXiv:1906.05409 (2019).

- [2] Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. “Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.” In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.

- https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d
