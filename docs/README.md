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

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/colormap.png?token=AGCBXXSWIDCCJZZ2YP5I3VK6USVVU "Severity Color")
corresponds to Severity 1, 2, 3, 4.

- Distribution of severity samples across the continental US.
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/map_usa.png?token=AGCBXXVTJESKAO74VIUBBGS6USVWU "Map of US Accidents")

- Crash occurance among each severity category. The frequency of the four levels of severity will play an important role in our analysis.

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/us_histogram.png?token=AGCBXXQL3CCHTMVH6YP7KH26TZAXY "Frequency of Severity in US")

- As well as the accident occurance for each state.
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/accidents_per_state.png?token=AGCBXXXYVDOT22PROW2EHVC6USVTO "Accident Counts for all States")

### Georgia

- Distribution of severity samples across the state of Georgia.
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/GA.png?token=AGCBXXUHAPRSODNNXTHI3326USVPY "Map of GA Accidents")

- Crash occurance among each severity category.

![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/moduleX_data/GA/ga_histogram.png?token=AGCBXXRC3OGJQ5VIS6UZFHK6USVRM "Frequency of Severity in GA")

# Approach

## What are you trying to do to tackle with your project motivation or problem?
As more and more Georgia drivers become aware of road conditions along their respective routes, there could be a significant reduction in the number of automotive accidents, injuries, and fatalities. Our team has used several predictive models to assess severity (from a scale of 1-4) for exclusively the roads in Georgia that can be used to evaluate driving conditions and take necessary precautions. 

## What have people already done?
In the study “A Perspective Analysis of Traffic Accidents using Data Mining Techniques” by S. Krishnaveni and Dr. Hemalatha, the researchers explored Naive Bayesian classifier, AdaBoostM1 Meta classifier, Random Forest Tree classifier, and PART Rule classifier to predict injury severity caused by traffic accidents in Hong Kong [5]. The research collected data based on accident (severity, weather, type of collision, road classification), vehicle (driver age, gender, manufacture date) , and casualty (location of casualty, degree of injury). As a result of this study, the Random Forest predictive model outperformed the other three models.

In our study, we have used relevant features such as severity, precipitation, weather condition, time of day, road type, and so forth to assess severity along Georgia roads. We used Principle Component Analysis as our dimension reduction technique on our dataset. Moreover, we have implemented Logistic Regression, Support Vector Machine, and Decision Tree classification models to see which model can predict severity most accurately.

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
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/module1_data/GA/correlation_original.png?token=AGCBXXVF7VUQRYKHOR5HIG26USV2O "Original Correlation")

- Applying PCA with 97% recovered variance resulted in the top 49 principal components.
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/module1_data/GA/pca_f194.png?token=AGCBXXQKFMBSBTXX6G6UU2C6USW26 "PCA scree plot")

- The original correlation is removed after performing PCA. This is confirmed by the diagonal line in the resulting correlation analysis indicating the selected principal components are orthogonal to one another and thereby linearly independent (ie. not correlated).
![alt text](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/module1_data/GA/correlation_pca.png?token=AGCBXXRRDJJF4FBFJPOXJ526USV3M "PCA Correlation")

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

-Penalty: Specifies the type of normalization used. The default value is l2.

-Inverse of regularization(C): Smaller values of this hyper-parameter indicates a stronger regularization. Default value is 1.0

-Random state : Seed used by the random number generator. Default value is None.

-Solver: Indicates which algorithm to use in the optimization problem. Default value is lbfgs.

-Max iter : max_iter represents maximum number of iterations taken for the solvers to converge a training process.


#### Results

Accuracy score = .527
![alt text](https://github.com/alexanderfache6/traffic-accident-weather-analysis/blob/master/code/Values%20vs%20Predictions.png)


### Support Vector Machine (SVM)

#### Description

SVM maps data into a high dimension space so that decision boundaries can distinguish between the different classes.

#### Implementation

Hyperparameters

- C (regularization)
  - 100
- <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a> (kernel coefficient)
  - 1

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

SVM struggled to fit onto the test after performing well on the training set, with 0.9997 and 0.479 accuracy respectively. An issue that was further researched is that SVM tends to work best for datasets consisting of fewer than 10,000 features. Our training and test sets were both greater and therefore may have caused intense overfitting due to an inappropriate selection of the number of support vectors.

### Gradient Boosting/Ensemble Learning using Decision Trees

#### Description

Gradient boosting combines small decision trees (relatively weak estimators) through a gradient descent algorithm rather than creating a single decision tree in order to produce a classification strong model that is robust to overfitting. Sk-learn has been implementing an experimental approach to gradient boosting using histograms to bin data and speed up calculations. This is the implementation we used.

#### Implementation

Hyperparameters:

```
- learning_rate
- max_iter
- max_leaf\_nodes
- min_samples\_leaf
- max_depth
```

#### Results
Results were first obtained with single iterations and some manual tuning of parameters. Further hyperparameter tuning was performed implementing sklearn.model_selection.GridSearchCV.
Results shown (for comparing both training and test sets to their respective ground truths): 

- Confusion Matrices
- Accuracy Score
- Prediction Score (for each individual label)
- F1 Score (for each individual label)

##### Single Run:
Hyperparameters for results shown:

- learning_rate: 0.1
- max_iter: 100
- max_leaf\_nodes: default=20
- min_samples\_leaf: 50
- max_depth: 8

Results:

![Single Run Results](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/Decision_singlerun.png?token=AKF5GLTSWS4H5L6R46EQRQC6UR45K)

##### GridSearchCV (Hyperparameter optimization):
Search Space explored:

- learning_rate: [0.05, 0.1, 0.15, 0.2]
- max_iter: [100, 500, 1000]
- max_leaf\_nodes: default=20
- min_samples\_leaf: [30, 50, 100]
- max_depth: [5, 6, 7, 8]

Due to time constaints, max leaf nodes was kept at default setting.

Results:

![Grid Search Best Results](https://raw.githubusercontent.com/alexanderfache6/traffic-accident-weather-analysis/master/code/Decision_gridsearch.png?token=AKF5GLSHBLKXYELRFW5QZ6S6UR5IM)

# Conclusion/Discussion

##### Overall Discussion

Overall, the project found some promise in its approach, but it is clear that perhaps more preprocessing or a different dataset is needed. Severity scores for traffic accidents were heavily skewed towards scores of either 1 or 2, which may have led to significant decreases in scoring metrics across all algorithms tested. However, each algorithm will be discussed and assessed on its own as well as compared/evaluated at the end of the discussion.

##### Gradient Boosting/Decision Trees

Gradient boosting is designed as a powerful combination of weak estimators that creates a model not as susceptible to overfitting as a standard decision tree. In this case, we can observe this through the relatively comparable accuracy, precision and F1 scores for training and test data. However, these scores remain fairly low. While some of these metrics tend to be harsh when looking at multilabel classification, the underlying bias of the traffic dataset towards severity 1 and 2 crashes (as well as an almost negligible amount of severity 0 scores) is a likely cause of the relatively low scoring metrics.

Further steps to improve the algorithm would include more directed hyperparameter tuning with a larger search space, as well as looking for ways to mitigate the skew of data (perhaps through stratified random sampling when constructing the training set to get more even numbers for each sample), and perhaps removing severity 0 traffic accidents entirely.

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
