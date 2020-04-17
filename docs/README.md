Note: This project is still in progress.

https://mahdi-roozbahani.github.io/CS46417641-spring2020/other/Scoring%20scheme-guidance.pdf

# Motivation
In a 2008 crash analysis report, the state of Georgia had an estimate of 342,534 traffic accidents. Out of which, 133,555 individuals were injured and 1,703 were dead. On an average, Georgia faces around 1,000 traffic accidents per day.

One explanation for higher crash rates in Georgia roads is that extreme road conditions due to weather (e.g. rain, snow, ice) create potential safety hazards. Such potential safety hazards include, but not limited to: driver(s) lose complete control of vehicle(s), improper lane change, or obstruction of visibility. 
The United States Department of Transportation Road Weather Management Program reports that annual averages from 2007-2016 show 15% of vehicle crashes occurred due to wet pavements with 10% due to rain, 4% due to snow, and 3% due to ice [1].

Eliminating weather conditions and associated factors is not possible, however, understanding relations between such conditions and crash risk could make drivers more aware of dangerous conditions. The following presents an analysis of US traffic accidents surveyed over the span of several years with the intention of developing a severity assessment model, ie. How do weather conditions impact crash damage?


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
As more and more Georgia drivers become aware of road conditions along their respective routes, there could be a significant reduction in the number of automotive accidents, injuries, and fatalities. Our team has used several predictive models to determine severity (from a scale of 1-4) for exclusively the roads in Georgia that can be used to evaluate driving conditions and take necessary precautions. 

## What have people already done?
In the study “A Perspective Analysis of Traffic Accidents using Data Mining Techniques” by S. Krishnaveni and Dr. Hemalatha, the researchers explored Naive Bayes Bayesian classifier, AdaBoostM1 Meta classifier, Random Forest Tree classifier, and PART Rule classifier to predict injury severity caused by traffic accidents in Hong Kong [5]. The research collected data based on accident (severity, weather, type of collision, road classification), vehicle (driver age, gender, manufacture date) , and causality (location of casualty, degree of injury). As a result of this study, the Random Forest predictive model outperformed the other three models.

In our study, we have used relevant features such as severity, precipitation, weather condition, time of day, road type, and so forth to predict severity along Georgia roads. We used Principle Component Analysis as our dimension reduction technique on our dataset. Moreover, we have implemented Logistic Regression, Support Vector Machine, and Decision Tree classification models to see which model can predict severity most accurately.

By implementing a predictive machine learning model fed with informative data, Georgia users (drivers) can explore the most dangerous locations along their commutes during extreme weather conditions to either avoid or take extra precautions. Our study can also be extended to locations beyond Georgia, but for short, we focused on this specific state to explore with.

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

Logistic regression is a regression technique employed to fit accident systems. Logistic regression techniques have been used to model probabilistic systems to predict future events. 

#### Implementation

```
too much. just highlight the important pieces. what is the sklearn function? how do you train it, get predictions?
hyperparameters are the parameters you need to optimize for, not the "constants": regularization (C)
```


Hyperparameters:

X, y: The first parameter is the dataset that is being selected to use

train_size: This parameter sets the size of the training dataset. 0.8 was used based on the ideal split 80:20 for training/testing

test_size: This parameter specifies the size of the testing dataset. 0.2 was used based on the ideal split 80:20 for training/testing

random_state: The default mode performs a random split using np.random. Was set to 0


#Assign the data:

df=df_state_dummy

#Set the target for the prediction:

target='Severity'


#Create arrays for the features and the response variable:

#set X and y:

y = df[target]

X = df.drop(target, axis=1)

#Split the data set into training and testing data sets:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=0, shuffle=True)

print('X train:', X_train.shape)

print('y train:', y_train.shape)

print('X test: ', X_test.shape)

print('y test: ', y_test.shape)

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

### Gradient Boosting/Ensemble Learning using Decision Trees

#### Description

- Gradient boosting combines small decision trees (relatively weak estimators) through a gradient descent algorithm rather than creating a single decision tree in order to produce a classification strong model that is robust to overfitting. Sk-learn has been implementing an experimental approach to gradient boosting using histograms to bin data and speed up calculations. This is the implementation we used.

#### Implementation

Hyperparameters

- learning_rate
- max_iter
- max_leaf\_nodes
- max_depth

#### Results

##### Single Run:
##### GridSearchCV (Hyperparameter optimization):

# Conclusion

```
- Were the experiments, results, and conclusion satisfactory?
- How did you evaluate your approach?
- What are the results?
- How do you compare your method to other methods?
````

# References

- [1] How do weather events impact roads? (2018). Federal Highway Administration. Retrieved from 
https://ops.fhwa.dot.gov/weather/q1roadimpact.htm

- [2] https://www.kaggle.com/sobhanmoosavi/us-accidents

- [3] Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, arXiv preprint arXiv:1906.05409 (2019).

- [4] Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. “Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.” In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.

- [5] Krishnaveni, S., & Hemalatha, M. (2011). A Perspective Analysis of Traffic Accident using Data Mining Techniques. International Journal of Computer Applications, 23(7), 40–48. doi: 10.5120/2896-3788
