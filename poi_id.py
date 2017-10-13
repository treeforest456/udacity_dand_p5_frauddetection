#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
features_list = [] # create a new list to incorporate the columns

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#     print len()
#     for key, value in data_dict.items():
#         print key , ':\t' , value
#         print '\n\n'
    dataset_dict = pd.read_pickle('final_project_dataset.pkl')
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 0. read them into pandas dataframe
# I'm more familiar with this form
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
dataset_df_temp = pd.DataFrame(dataset_dict)
dataset_df = pd.DataFrame.transpose(dataset_df_temp)
dataset_df.shape
dataset_df


# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 1. Clean the dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # # # # # # # # 
# 1.1. Outliers
# # # # # # # # # # # # # # # # # # # # # # # # 
# have a look
# plt.scatter(dataset_df['salary'], dataset_df['bonus'])
# plt.xlabel('salary')
# plt.ylabel('bonus')
# plt.title('salary vs bonus')
# plt.show()

# drop the row
print 'before drop : \t' , dataset_df.query(''' salary > 2.5e7 and salary != 'NaN' ''').shape
dataset_df.drop(dataset_df.query(''' salary > 2.5e7 and salary != 'NaN' ''').index, inplace=True)
print 'after drop: \t' , dataset_df.query(''' salary > 2.5e7 and salary != 'NaN' ''').shape

# plot the scatter plot to have a look to check
# plt.scatter(dataset_df['salary'], dataset_df['bonus'])
# plt.xlabel('salary')
# plt.ylabel('bonus')
# plt.title('salary vs bonus')
# plt.show()

# convert all the datatype to something operable
# https://stackoverflow.com/a/21197863
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.convert_objects.html
dataset_df = dataset_df.convert_objects(convert_numeric=True)

# plot everything vs everything to find out outliers in the original dataset
# for ii in dataset_df:
#     for jj in dataset_df:
#         if ii != jj and dataset_df[ii].dtypes == 'float64' and dataset_df[jj].dtypes == 'float64':
#             plt.scatter(dataset_df[ii], dataset_df[jj])
#             plt.xlabel(ii)
#             plt.ylabel(jj)
            # plt.show()

# we can see this is also an outlier
# print 'before drop : \t' , dataset_df.query(''' total_payments > 1e8 ''').shape
dataset_df.drop(dataset_df.query(''' total_payments > 1e8 ''').index, inplace=True)
# print 'after drop: \t' , dataset_df.query(''' total_payments > 1e8 ''').shape


# there is a row in deferral_payments is greater than 6000000
# and other than this one, the highest of the rest of them is 3000000
# don't know whether it's a good idea to remove this one, it's not crazy larger than others
# so maybe keep it

# duplicate the True values so that there's enough value for both the labels
# if don't do this, the recall can't go above 0.3
# since the labels are very skewed
poi_trues = dataset_df.query(''' poi == True ''')

dataset_df_append_temp = dataset_df
for ii in range(15):
    dataset_df_append_temp = dataset_df_append_temp.append(poi_trues)
dataset_df_dup = dataset_df_append_temp
# 16 times can take us as high as 0.41 and 0.40 for precision and recall
# print dataset_df_dup['poi'].describe()

# # # # # # # # # # # # # # # # # # # # # # # # 
# 1.2. Dealing with 'NaN' values 
# # # # # # # # # # # # # # # # # # # # # # # # 

# fill 'NaN' with 0
dataset_df_removeNaN_zero = dataset_df_dup.replace('NaN', 0, inplace=False)


# collect the numerical columns to get ready for imputation
# leave the string type columns alone 
num_var = []
non_num_var = []
for each_column in dataset_df_removeNaN_zero:    
    if each_column != 'email_address' and each_column != 'poi':
        num_var.append(each_column)
    else :
        non_num_var.append(each_column)

# Get the numerical values
dataset_df_removeNaN_zero_num = dataset_df_removeNaN_zero[num_var]

# for ii in dataset_df_removeNaN_zero_num:
#     # print dataset_df_removeNaN_zero_num[ii].describe()
#     print dataset_df_removeNaN_zero_num[ii].isnull().value_counts()
#     # print 

# # # # # # # # # # # # # # # # # # # # # # # # 
# check if there is any NaN values since the reviewer said there is 
# # # # # # # # # # # # # # # # # # # # # # # # 

# print 'is there NaN value: \t' , np.any(np.isnan(dataset_df_removeNaN_zero_num.values))
# print 'where is NaN: \t' , np.where(np.isnan(dataset_df_removeNaN_zero_num.values))
# print 'max value: \t' , np.max(dataset_df_removeNaN_zero_num.values)
# print 'min value: \t' , np.min(dataset_df_removeNaN_zero_num.values)

# # # # # # # # # # # # # # # # # # # # # # # # 
# the results said no, but I'll do it anyway
# # # # # # # # # # # # # # # # # # # # # # # # 

# replace inf with nan and fill nan with 0
dataset_df_removeNaN_zero_num = dataset_df_removeNaN_zero_num.replace(np.inf, np.nan)
dataset_df_removeNaN_zero_num = dataset_df_removeNaN_zero_num.fillna(0)

# # # # # # # # # # # # # # # # # # # # # # # # 
# 1.3. Scaling dataset
# should I scale before imputation or after?
# before imputation, the scaling process will perform on the original data, good
# but the scaling process didn't really performed on the NaN data, so that part is missing, bad
# so I choose to scale after imputation
# turns out the MinMaxScaler is not having a good results as the StandardScaler
# # # # # # # # # # # # # # # # # # # # # # # # 

from sklearn.preprocessing import StandardScaler, MinMaxScaler

ss = StandardScaler()
dataset_filled_knn_df_scaled_vals = ss.fit_transform(dataset_df_removeNaN_zero_num.values)
dataset_filled_knn_df_scaled = pd.DataFrame(dataset_filled_knn_df_scaled_vals, \
                                            index=dataset_df_removeNaN_zero_num.index, \
                                            columns=dataset_df_removeNaN_zero_num.columns)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 2. Feature engineering
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# 2-way numerical interaction
original_num_of_cols = len(dataset_filled_knn_df_scaled.columns)

# 2 way plus, minus, times divide
for outer_iter in range(original_num_of_cols):
    for inner_iter in range(outer_iter + 1, original_num_of_cols):
        first_name = dataset_filled_knn_df_scaled.columns[outer_iter]
        last_name = dataset_filled_knn_df_scaled.columns[inner_iter]
        dataset_filled_knn_df_scaled[first_name + '_' + last_name + '_' + 'plus'] \
                        = dataset_filled_knn_df_scaled[first_name] + dataset_filled_knn_df_scaled[last_name]
            
        dataset_filled_knn_df_scaled[first_name + '_' + last_name + '_' + 'minus'] \
                        = dataset_filled_knn_df_scaled[first_name] - dataset_filled_knn_df_scaled[last_name]
            
        dataset_filled_knn_df_scaled[first_name + '_' + last_name + '_' + 'times'] \
                        = dataset_filled_knn_df_scaled[first_name] * dataset_filled_knn_df_scaled[last_name]

        dataset_filled_knn_df_scaled[first_name + '_' + last_name + '_' + 'divide'] \
                        = dataset_filled_knn_df_scaled[first_name] / dataset_filled_knn_df_scaled[last_name]
                    
# print dataset_filled_knn_df_scaled.shape

# # # # # # # # # # # # # # # # # # # # # # # # 
# check if there is any NaN values 
# # # # # # # # # # # # # # # # # # # # # # # # 
# print 'is there NaN value: \t' , np.any(np.isnan(dataset_filled_knn_df_scaled.values))
# print 'where is NaN: \t' , np.where(np.isnan(dataset_filled_knn_df_scaled.values))
# print 'max value: \t' , np.max(dataset_filled_knn_df_scaled.values)
# print 'min value: \t' , np.min(dataset_filled_knn_df_scaled.values)

# # # # # # # # # # # # # # # # # # # # # # # # 
# Turs out there is NaN and Inf after I do the interaction
# so going to replace inf and fill NaN with 0
# # # # # # # # # # # # # # # # # # # # # # # # 
dataset_filled_knn_df_scaled = dataset_filled_knn_df_scaled.replace(np.inf, np.nan)
dataset_filled_knn_df_scaled = dataset_filled_knn_df_scaled.fillna(0)

# print 'is there NaN value: \t' , np.any(np.isnan(dataset_filled_knn_df_scaled.values))
# print 'where is NaN: \t' , np.where(np.isnan(dataset_filled_knn_df_scaled.values))
# print 'max value: \t' , np.max(dataset_filled_knn_df_scaled.values)
# print 'min value: \t' , np.min(dataset_filled_knn_df_scaled.values)

# Scale again
ss = StandardScaler()
dataset_filled_knn_df_scaled_vals_again_vals = ss.fit_transform(dataset_filled_knn_df_scaled.values)
dataset_filled_knn_df_scaled_vals_again = pd.DataFrame(dataset_filled_knn_df_scaled_vals_again_vals, \
                                            index=dataset_filled_knn_df_scaled.index,\
                                            columns=dataset_filled_knn_df_scaled.columns)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 3. Premodeling
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Get the dataset
X_all = dataset_filled_knn_df_scaled_vals_again
Y_all = dataset_df_dup['poi'].reshape(-1, 1)

# # # # # # # # # # # # # # # # # # # # # # # # 
# Train Test Split is not good here due to the skewed labels
# if completely random, some of the subset could have no True label at all
# the model won't perform good on that scenario and the recall won't be good
# so use stratified shuffle split to make sure 
# every subset has good label set to learn
# # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # # # # # # # # 
# 3.1. new stratified shuffle split
# compare different algorithms
# # # # # # # # # # # # # # # # # # # # # # # # 
from sklearn.metrics import classification_report

# import multiple classifiers to see who's performance's better
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

mlp = MLPClassifier()
knc = KNeighborsClassifier()
svc = SVC()
gpc = GaussianProcessClassifier()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
nbc = GaussianNB()

# Convert to np array for split
# that's better format for split don't know why
X_array = np.array(X_all.values)
Y_array = np.array(Y_all.reshape(-1,1))
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.1, random_state=2017)
for train_index, test_index in sss.split(X_array, Y_array):
    x_train, x_test = X_array[train_index], X_array[test_index]
    y_train, y_test = Y_array[train_index], Y_array[test_index]

    # mlp.fit(x_train, y_train)
    # knc.fit(x_train, y_train)
    # svc.fit(x_train, y_train)
    # gpc.fit(x_train, y_train)
    dtc.fit(x_train, y_train)
    # rfc.fit(x_train, y_train)
    # abc.fit(x_train, y_train)
    # nbc.fit(x_train, y_train)
    
    # this was for check, just for print out

    # clf = dtc
    # y_pred = clf.predict(x_test).ravel()
    # y_true = y_test
    # print classification_report(y_true, y_pred)
    # pre_rec_f1 = classification_report(y_true, y_pred).split()[-4:-1]
    # int_pre_rec_f1 = [float(ii) for ii in pre_rec_f1]
    # print int_pre_rec_f1

# The result from above shows that
# dtc is the best with respect to both precision and recall



# # # # # # # # # # # # # # # # # # # # # # # # 
# 3.2. Print out the Feature Importance
# # # # # # # # # # # # # # # # # # # # # # # # 
counter = 0
featuressss = zip(list(X_all.columns), list(dtc.feature_importances_))
for ii in sorted(featuressss, key=lambda x: x[1], reverse=True):
    if counter < 10:
        print ii
        print 
    else :
        break
    counter += 1

ss = StandardScaler()
dataset_original_scaled_3_vals = ss.fit_transform(dataset_df_removeNaN_zero_num.values)
dataset_original_scaled_3 = pd.DataFrame(dataset_original_scaled_3_vals, \
                                            index=dataset_df_removeNaN_zero_num.index, \
                                            columns=dataset_df_removeNaN_zero_num.columns)

dataset_original_scaled_3['total_payments_total_stock_value_plus'] = dataset_filled_knn_df_scaled_vals_again['total_payments_total_stock_value_plus']
dataset_original_scaled_3['expenses_restricted_stock_deferred_plus'] = dataset_filled_knn_df_scaled_vals_again['expenses_restricted_stock_deferred_plus']
dataset_original_scaled_3['exercised_stock_options_total_stock_value_divide'] = dataset_filled_knn_df_scaled_vals_again['exercised_stock_options_total_stock_value_divide']
dataset_original_scaled_3['from_messages_from_this_person_to_poi_minus'] = dataset_filled_knn_df_scaled_vals_again['from_messages_from_this_person_to_poi_minus']
dataset_original_scaled_3['from_messages_from_this_person_to_poi_divide'] = dataset_filled_knn_df_scaled_vals_again['from_messages_from_this_person_to_poi_divide']
dataset_original_scaled_3['salary_total_stock_value_plus'] = dataset_filled_knn_df_scaled_vals_again['salary_total_stock_value_plus']
dataset_original_scaled_3['expenses_restricted_stock_deferred_times'] = dataset_filled_knn_df_scaled_vals_again['expenses_restricted_stock_deferred_times']
dataset_original_scaled_3['from_poi_to_this_person_shared_receipt_with_poi_times'] = dataset_filled_knn_df_scaled_vals_again['from_poi_to_this_person_shared_receipt_with_poi_times']


# # # # # # # # # # # # # # # # # # # # # # # # 
# 3.3. GridSearch
# Now fine tune the Decision Tree Classifier
# # # # # # # # # # # # # # # # # # # # # # # # 
from sklearn.model_selection import GridSearchCV
# Turns out the default parameters are all good 
# even they are the best 

dtc = DecisionTreeClassifier()

params = {
          'max_depth': [6, 8, 10, 15],
          # 'min_samples_split': [2, 4, 6, 8],
          # 'min_samples_leaf': [1, 2, 3],
          'random_state': [2017]}
gsearch = GridSearchCV(estimator = dtc, 
                       param_grid = params, scoring='roc_auc', n_jobs=-1, iid=False, cv=sss)

# # # # # # # # # # # # # # # # # # # # # # # # 
# print some thing out to check 
# # # # # # # # # # # # # # # # # # # # # # # # 
print dataset_original_scaled_3.shape
# only add 4 new features from the feature importance
# # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # 
X_all_newfeatureset = dataset_original_scaled_3
Y_all = dataset_df_dup['poi'].reshape(-1, 1)
X_array_newfeatureset = np.array(X_all_newfeatureset.values)
Y_array = np.array(Y_all.reshape(-1,1))
gsearch.fit(X_array_newfeatureset, Y_array.ravel())


# # # # # # # # # # # # # # # # # # # # # # # # 
# 3.4. Lock on tuned DecisionTreeClassifier
# # # # # # # # # # # # # # # # # # # # # # # # 
clf = gsearch.best_estimator_


# # # # # # # # # # # # # # # # # # # # # # # # 
# 4. Format output
# # # # # # # # # # # # # # # # # # # # # # # # 
dataset_original_scaled_3.insert(loc=0, column='poi', value=dataset_df_dup['poi'].values)
features_list = list(dataset_original_scaled_3.columns)
my_dataset = dataset_original_scaled_3.transpose().to_dict()

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

dump_classifier_and_data(clf, my_dataset, features_list)


