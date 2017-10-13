# udacity_dand_p5_frauddetection
udacity_dand_p5_frauddetection

## This is a project from Udacity Data Analyst Nanodegree

In this project, I use machine learning skills built an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

### Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

### Some relevant files
```poi_id.py``` : Starter code for the POI identifier, you will write your analysis here. You will also submit a version of this file for your evaluator to verify your algorithm and results. 

```final_project_dataset.pkl``` : The dataset for the project, more details below. 

```tester.py``` : When you turn in your analysis for evaluation by Udacity, you will submit the algorithm, dataset and list of features that you use (these are created automatically in poi_id.py). The evaluator will then use this code to test your result, to make sure we see performance that’s similar to what you report. You don’t need to do anything with this code, but we provide it for transparency and for your reference. 

* There also some files in ```tools``` folder that are needed to convert files to .pkl format for project review.

### Steps:
1. Preprocessing
	* Outliers
	* Filling NaN
	* Scaling
2. Feature Engineering
	* 2-way numerical interaction( + - * ÷ )
	* Scaling
3. Modeling
	* Model comparing
	* Stratified splitting
	* Feature selection
	* GridSearchCV
	* Parameter tuning

### Result:
Used Decision Tree Classifier
Got the following scores:
```python
	Accuracy: 0.81747	Precision: 0.31605	Recall: 0.31700	F1: 0.31653	F2: 0.31681
	Total predictions: 15000	True positives:  634	False positives: 1372	False negatives: 1366	True negatives: 11628
```
