## Documentation:

### Things about Data Points:
The original total data point is 146 individuals, each of them has 21 features. There are 18 individuals of the 146 are True POIs.

I drop email and poi columns because poi is target variable, and don't want to analyze email column.

So the shape became (146, 19)

During dropping ouliers process, I dropped 2 rows, detailed in the corresponding question below.

Now the shape is (144, 19)

I find the labels in this dataset is skewed, 127 of 144 are False, and only 17 of them are True. It's not good for the model to get a good precision and recall score. So I append the True poi rows to the dataset 15 times.

144 + 15 * 17 = 399

Now the shape is (399, 19)

I create many new features using these 19 columns, after that, I have 703 columns in total.

Now, the total number of data points is 399 rows and 703 columns.
272 rows of them are POIs.

Shape of it was (399, 703)

After I trained the Decision Tree Classifier and got the feature importance, I picked the top 8 features to add to the original dataset. After that, the final shape of the dataset is (399, 27).

So the final dataset has the shape of (399, 27) and then it goes into the GridSearchCV function to get the classifier as the final result.


### Questions:

Q1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

A1: The goal of this project is using the Enron email dataset to train several machine learning algorithms to make predictions. Machine learning algorithms can learn from the training data that we have, and predict on the new data that we have never seen. The dataset includes many numerical data, such like salary, bonus etc. The target is the 'poi' column, whether a True or False. There were some outliers in the raw dataset, I used matplotlib package to show the scatter plot and found the outliers, then just remove the datapoint.
One of the outlier is called 'TOTAL', the other one called 'LAY KENNETH L'.

* The 'TOTAL' is out because of salary > 2.5e7.
The code I use to drop 'TOTAL' is :
```python
print 'before drop : \t' , dataset_df.query(''' salary > 2.5e7 and salary != 'NaN' ''').shape
dataset_df.drop(dataset_df.query(''' salary > 2.5e7 and salary != 'NaN' ''').index, inplace=True)
print 'after drop: \t' , dataset_df.query(''' salary > 2.5e7 and salary != 'NaN' ''').shape
```
And the result is :
```python
before drop :   (1, 21)
after drop:     (0, 21)
```

* The 'LAY KENNETH L' is out because of total_payments > 1e8.
The code I use to drop 'LAY KENNETH L' is :
```python
print 'before drop : \t' , dataset_df.query(''' total_payments > 1e8 ''').shape
dataset_df.drop(dataset_df.query(''' total_payments > 1e8 ''').index, inplace=True)
print 'after drop: \t' , dataset_df.query(''' total_payments > 1e8 ''').shape
```
And the result is :
```python
before drop :   (1, 21)
after drop:     (0, 21)
```
We can see that from the plot.

### =================

Q2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

A2. I used all the numerical features. I use as many features as possible. I performed stand scaling, which is package in sklearn. The reason I scale the data is to make it easier for the algorithms to do the fitting. I create 2-way interaction between numerical features, such as plus, minus, times. We can know more about the characteristic about each persons by know what the sum of salary and bonus one person has in total. There could be a person who has little salary, but got much bonus. By creating the salary-plus-bonus, we know more about a single person. The top 3 features sorted by feature_importance_ are the following:

* ('total_payments_total_stock_value_plus', 0.48410274833842282)
* ('expenses_restricted_stock_deferred_times', 0.19291029709567528)
* ('exercised_stock_options_total_stock_value_divide', 0.11865455869071764)

The feature importances are from the clssifiers built-in function, and I wrote following code to show them:
```python
 featuressss = zip(list(dataset_filled_knn_df_scaled.columns), list(dtc.feature_importances_))
for ii in sorted(featuressss, key=lambda x: x[1], reverse=True):
    print ii
```
The total number of features I use is 703. I append the True values to the dataset several times to make the labels of the training set less skewed, so there are 399 rows.

Because I print out the Top 10 features from feature importance, then I hand pick the Top some from them, added to the dataset. Now I have a dataset with 399 rows and 27 columns. 

The Top 8 are:
- *total_payments_total_stock_value_plus*
- *expenses_restricted_stock_deferred_plus*
- *exercised_stock_options_total_stock_value_divide*
- *from_messages_from_this_person_to_poi_minus*
- *from_messages_from_this_person_to_poi_divide*
- *salary_total_stock_value_plus*
- *expenses_restricted_stock_deferred_times*
- *from_poi_to_this_person_shared_receipt_with_poi_times*

Without:
```python
    Accuracy: 0.79073   Precision: 0.19788  Recall: 0.18650 F1: 0.19202 F2: 0.18867
    Total predictions: 15000    True positives:  373    False positives: 1512   False negatives: 1627   True negatives: 11488
```

With: Top 4 new features
```python
    Accuracy: 0.81107   Precision: 0.28746  Recall: 0.28200 F1: 0.28470 F2: 0.28308
    Total predictions: 15000    True positives:  564    False positives: 1398   False negatives: 1436   True negatives: 11602
```

With: Top 6 new features
```python
    Accuracy: 0.81420   Precision: 0.30116  Recall: 0.29800 F1: 0.29957 F2: 0.29863
    Total predictions: 15000    True positives:  596    False positives: 1383   False negatives: 1404   True negatives: 11617
```

With: Top 8 new features
```python
    Accuracy: 0.81747   Precision: 0.31605  Recall: 0.31700 F1: 0.31653 F2: 0.31681
    Total predictions: 15000    True positives:  634    False positives: 1372   False negatives: 1366   True negatives: 11628
```


The new features can provide some more information about the dataset to help the algorithm understand the dataset better in order to make a better prediction. Creating more features can increase the complexity of the dataset and describe each datapoint in a higher dimension to provide more information for the algorithm also can prevent the model from overfitting the original dataset. And overfitting is something we don't want to happen in our prediction process. 
In these 8 features we can see some of them are the sum of two variables, some are the difference, product, and quotient of two variables. These values can show us some other aspects of the dataset, such as the ratio of from_messages and from_this_person_to_poi where we can get from 'from_messages_from_this_person_to_poi_divide' feature. These features create some new windows about how to observe the dataset and also some new characteristics for each individuals. Those are all the things the model could take advantage of.

The final feature set is a combination of the original 19 columns plus the 8 new features I mentioned above, which make the final number of columns equal to 27.

### =================

Q3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

A3. I end up using Decision Tree Classifier, because it gave the best performance both on precision and recall. Other algorithms either low on precision or recall. However, at first, the precision and recall are not over 0.3, they were both around 0.24. I appended the True values to the dataset, to make the class less skewed to make it easier for the classifier. The Multi-Layer Perceptron Classifier also gave the performance both around 0.22. I tried multiple classifiers for this problems, such as RandomForest, SVC, AdaBoost and etc.
Here are the some of the scores:
```python
Classifier                  Precision         Recall
NaiveBayes                     0.14             0.66
MultilayerNN                   0.365            0.328
AdaBoost                       0.37             0.189
DecisionTree                   0.41             0.404
```

### =================

Q4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]


A4. Tuning parameter means you have to make your model fit the dataset just right. The model could underfit or overfit the training set if the parameter are not good enough. I used GridSearchCV to tune the parameters of decision tree classifier and the parameters I tuned are the following: max_depth, min_samples_split, min_samples_leaf and random_state.

### =================

Q5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

A5. Validation is the process you validate the performance of your model. You could use the same data points do the validation and test if you do it wrong. In StratifiedShuffleSplit, I use 10% of the whole dataset as test set. In validation process, you use some of the known dataset out of training set as a validation data to have a look at the performance of the model after training. If the performance is not good enough, you choose another model or tune the parameters of this model, and then do training and validation process. Untill you find a good model with good parameters in terms of good performance in the validation set. Now you have your model good enough to further predict on the data the model has never seen.

### =================

Q6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

A6. Accuracy, precision and recall. Precision means the True Positive of all the Positive prediction. Recall means the True Positive of all the actual Positive. Decision Tree Classifier got 0.81747, 0.31605, 0.317 respectively.

Accuracy: 0.81747   Precision: 0.31605  Recall: 0.31700 F1: 0.31653 F2: 0.31681
Total predictions: 15000    True positives:  634    False positives: 1372   False negatives: 1366   True negatives: 11628

Which means the model can point out 634 POIs while there are 634 + 1366 POIs out there. Means if there are 2000 bad guys out there, our model can correctly send 634 of them to the prison, and 1366 are still out there doing bad things. For the precision, it means the model think 634 + 1372 = 2006 people are bad guys and were sent to prison, there only 634 of them are real bad, the other 1372 people are actually good guys.
So the precision means for all the 2006 people the model thought are bad, only 31.6% of them are real criminals. On the other hand, recall means from all the 2000 criminals, the model can only point out 31.7% of them as bad guys.











