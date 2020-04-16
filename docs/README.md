Note: This project is still in progress.

https://mahdi-roozbahani.github.io/CS46417641-spring2020/other/Scoring%20scheme-guidance.pdf

# Motivation
In a 2019 Business Insider report, the city of Atlanta is reported to be ranked as number 11 on the most traffic congested cities in the United States [CITATION]. Unfortunately, thousands of Atlanta commuters are plagued with motor vehicle collisions every year in various traffic conditions. According to a published 2016 traffic report of Fulton County, Atlanta has faced an estimate of 60,984 automotive accidents with 12,875 injuries. One explanation for higher crash rates in Atlanta roads is that extreme road conditions due to weather (e.g. rain, snow, ice) create potential safety hazards. Such potential safety hazards include, but not limited to: driver(s) lose complete control of vehicle(s), improper lane change, or obstruction of visibility. The United States Department of Transportation Road Weather Management Program reports that annual averages from 2007-2016 show 15% of vehicle crashes occurred due to wet pavements with 10% due to rain, 4% due to snow, and 3% due to ice [1].

Eliminating weather conditions and associated factors is not possible, however, understanding relations between such conditions and crash risk could make drivers more aware of dangerous conditions. The following presents an analysis of US traffic accidents surveyed over the span of several years with the intention of developing a severity assessment model, ie. how do weather conditions impact crash damage. 

````
- was the motivation clear? X
- what is the problem? X
- why is it important and why we should care? X
````

# Dataset

````
- Were the dataset and approach used effectively?
- How did you get your dataset? X
- What are its characteristics (e.g. number of features, # of records, temporal or not, etc.) X
````

## [US Accidents](https://www.kaggle.com/sobhanmoosavi/us-accidents) 

The dataset used for this project was found on Kaggle and put together by [3]-[4]. It contains 3.0 million records of spatial-temporal traffic accidents across 49 US states from February 2016 to December 2019. Among these records, variables such as time of day, latitute/longitude, weather conditions, road features were collected. This section summarizes the dataset's features and provides additional insight to its organization.

### Features

Shown below are the 49 original features each identified by their keyword as saved in the corresponding Pandas DataFrame:

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| - | - | - | - | - | - | - | - | - | - |
| ID | Source | TMC | Severity | Start_Time | End_Time | Start_Lat | Stop_Lng | End_Lat | End_Lng |
| 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
| Distance(mi) | Description | Number | Street | Side | City | County | State | Zipcode | Country |
| 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
| Timezone | Airport_Code | Weather_Timestamp | Temperature(F) | Wind_Chill(F) | Humidity(%) | Pressure(in) | Visibility(mi) | Wind_Direction | Wind_Speed(mph) |
| 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
| Precipitation(in) | Weather_Condition | Amenity | Bumpy | Crossing | Give_Way | Junction | No_Exit | Railway | Roundabout |
| 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
| Station | Stop | Traffic_Calming | Traffic_Signal | Turning_Loop | Sunrise_Sunset | Civil_Twilight | Nautical_Twilight | Astronomical_Twilight |

Several of the features have incomplete values or categorical values and will need to be cleaned up during preprocessing.

### United States

First we consider the distribution of samples across the entire dataset noting the following color map to indicate the four levels of crash severity that will be used as our supervised labels:

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/colormap.png?token=AGCBXXWRWQFSQK6CDVVVR5S6TNUNM "Severity Color")
corresponds to Severity 1, 2, 3, 4.

- Distribution of severity samples across the continental US.
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/map_usa.png?token=AGCBXXR5JQCTMUPC43ZJ3AS6TNULW "Map of US Accidents")

- Crash occurance among each severity category. The frequency of the four levels of severity will play an important role in our analysis.

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/us_histogram.png?token=AGCBXXQL3CCHTMVH6YP7KH26TZAXY "Frequency of Severity in US")

- As well as the accident occurance for each state.
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/accidents_per_state.png?token=AGCBXXQBYLPFREZ3FDQZ7BC6TZATM "Accident Counts for all States")

### Georgia

- Distribution of severity samples across the state of Georgia.
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/GA.png?token=AGCBXXSWMABYG3ZU767G5OS6TNUCY "Map of GA Accidents")

- Crash occurance among each severity category.

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/ga_histogram.png?token=AGCBXXVQ4BSPAE3CYC2KFNS6TZAVC "Frequency of Severity in GA")

# Approach

## What are you trying to do to tackle with your project motivation or problem?
By implementing a real-time updating machine learning model fed with informative data, users (drivers) can explore the most dangerous locations along their commutes during extreme weather conditions to either avoid or take extra precautions. As more and more Atlanta drivers become aware of road conditions along their respective routes, there could be a significant reduction in the number of automotive accidents, injuries, and fatalities. Our team has developed a risk assessment for regions that drivers can use to evaluate driving conditions and take necessary precautions.

## What have people already done?
We have used relevant features such as precipitation, weather condition, time of day, road type, severity and so forth to calculate risk scores along Atlanta roads. WILL COME BACK TO THIS AFTER GROUP DISCUSSION MEETING

````
- Why do you think your approach can effectively solve your problem?
- What is new in your approach?
````

# Modules

## 1 Feature Extraction, Dimensionality Reduction, Feature Ranking

### Preprocessing

During preprocessing, the data set is first cleaned up. This means:

  - filling in missing categorical data with the mode of that feature
  - filling in missing numerical data with the medaian of that feature
  - cleaning up date time objects by splitting into year, month, day, hour, minute, second attributes
  - replace True/False data with 1/0
  - apply one-hot encoding for categorical data
    - ex. Sunrise_Sunset = {Day, Night}. Turn into Sunrise_Sunset_Day = {True, False}, Sunrise_Sunset_Night = {True, False}

### Principle Component Analysis (PCA)

- Aims to select principal components in Z space to attain the largest possible variance.
- Reduces dimensionality of data thereby reducing complexity.

- For the original data set, each feature has some correlation/dependency on other features.
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/module1_data/GA/correlation_original.png?token=AGCBXXQBKWDUN7AHONNL5IK6TNUPK "Original Correlation")

- The original correlation is removed after performing PCA. This is confirmed by the diagonal line in the resulting correlation analysis indicating the selected principal components are orthogonal to one another and thereby linearly independent (ie. not correlated).
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/module1_data/GA/correlation_pca.png?token=AGCBXXTXJNVRX6J2XDXBAUK6TNUQK "PCA Correlation")

## 2 Supervised Learning

### Logistic Regression

#### Description

- 1-2 sentence description

#### Implementation

Hyperparameters

- describe
- show key code lines

#### Results

- images!!!

### Support Vector Machine (SVM)

#### Description

SVM maps data into a high dimension space so that decision boundaries can distinguish between the different classes.

#### Implementation

Hyperparameters

- C (regularization)
  - ...
- <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a> (kernel coefficient)
  - ...

Parameters

- kernel (kernel type)
  - 'rbf' (radial based function)


```
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=c, gamma=g).fit(X_train, y_train)

score_train = svm.score(X_train, y_train)
score_test = svm.score(X_test, y_test)
```

#### Results

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/module2_data/GA/SupportVectorMachines/SVM_.png?token=AGCBXXSPGX2KQUD65FCI5YC6TTXY4 "C vs Gamma Accuracy")

### Decision Trees

#### Description

- 1-2 sentence description

#### Implementation

Hyperparameters

- describe
- show key code lines

#### Results

- images!!!

# Conclusion

```
- Were the experiments, results, and conclusion satisfactory?
- How did you evaluate your approach?
- What are the results?
- How do you compare your method to other methods?
````

# References

- [1] “How Do Weather Events Impact Roads?”, 2018

- [2] https://www.kaggle.com/sobhanmoosavi/us-accidents

- [3] Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, arXiv preprint arXiv:1906.05409 (2019).

- [4] Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. “Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.” In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.
