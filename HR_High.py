import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

hr_df= pd.read_csv("C:/Users/ssn/Desktop/capstone/train_HR.csv")
hr_test=pd.read_csv("C:/Users/ssn/Desktop/capstone/test_HR.csv")

hr_df.head(5)

hr_df.info()

# to find the null values in each column
hr_df.isnull().sum()
hr_test.isnull().sum()
#education               2409
#previous_year_rating    4124

#renaming column name
hr_df.rename(columns={'KPIs_met >80%': 'KPI','awards_won?' : 'awards'}, inplace=True)

#outliers detections
outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

detect_outlier(hr_df['age'])
#graph
sns.set(style="whitegrid")
ax = sns.boxplot(x=hr_df["age"])



#vif cal

vif_df = hr_df._get_numeric_data() #drop non-numeric cols

def vif_cal(input_data, dependent_col):
    import statsmodels.formula.api as sm
    x_vars = input_data.drop([dependent_col],axis=1)
    xvar_names = x_vars.columns
    for i in range(0,len(xvar_names)):
        y = x_vars[xvar_names[i]]
        x = x_vars[xvar_names.drop(xvar_names[i])]
        rsq = sm.ols("y~x",x_vars).fit().rsquared
        vif = round(1/(1-rsq),2)
        print(xvar_names[i], "VIF: ", vif)
        
vif_cal(vif_df, 'is_promoted')

#age and length of service highly collerated so drop age.

hr_df['education'].mode()[0]


#for the missing values
hr_df['education'][hr_df['region'] == 'region_10'].isnull().sum()
hr_df['education'][hr_df['region'] == 'region_10'].mode()[0]
hr_df['education'][hr_df['region'] == 'region_10'].isnull()

hr_df['previous_year_rating'].mode()


#find the mode based upon region
hr_df.groupby(['region'])['education'].agg(pd.Series.mode).to_frame()
hr_df.groupby(['department'])['previous_year_rating'].agg(pd.Series.mode).to_frame()



hr_test.groupby(['region'])['education'].agg(pd.Series.mode).to_frame()
hr_df.groupby(['department'])['previous_year_rating'].agg(pd.Series.mode).to_frame()

#filling the missing values
hr_df['education'][hr_df['region'] == 'region_10']=hr_df['education'][hr_df['region'] == 'region_10'].fillna(hr_df['education'][hr_df['region'] == 'region_10'].mode()[0])
hr_df['education'][hr_df['region'] != 'region_10']=hr_df['education'][hr_df['region'] != 'region_10'].fillna(hr_df['education'][hr_df['region'] != 'region_10'].mode()[0])

hr_df['previous_year_rating']=hr_df['previous_year_rating'].fillna(hr_df['previous_year_rating'].mode()[0])

hr_df['education']=hr_df['education'].fillna("Bachelor's")

#filling the missing values for test data
hr_test['education'][hr_test['region'] == 'region_1']=hr_test['education'][hr_test['region'] == 'region_1'].fillna("Master's & above")
hr_test['education'][hr_test['region'] == 'region_10']=hr_test['education'][hr_test['region'] == 'region_10'].fillna("Master's & above")
hr_test['education'][hr_test['region'] == 'region_4']=hr_test['education'][hr_test['region'] == 'region_4'].fillna("Master's & above")

hr_test['education']=hr_test['education'].fillna("Bachelor's")

hr_test['previous_year_rating']=hr_test['previous_year_rating'].fillna(hr_test['previous_year_rating'].mode()[0])

#dropping the columns
del hr_df['age']
del hr_df['length_of_service']
del hr_df['recruitment_channel']
del hr_df['gender_f']
del hr_df['region']

#dropping the columns in test
del hr_test['age']
del hr_test['length_of_service']
del hr_test['recruitment_channel']
del hr_test['gender_f']
del hr_test['region']

hr_df['work_fraction'] =hr_df['age'] - hr_df['length_of_service'] 

hr_test['work_fraction'] =hr_test['age'] - hr_test['length_of_service'] 



#converting categorical data into numbers

hr_df=pd.get_dummies(hr_df,columns=['department','education','gender'])

#converting categorical data into numbers in test data
hr_test=pd.get_dummies(hr_test,columns=['department','education','gender'])


#to find the list of columns
list(hr_df.columns.values)

list(hr_test.columns.values)




#splitting the dep and indep of the train data
Y = hr_df.is_promoted
X = hr_df.drop(['is_promoted','employee_id'],1)

#to find the accuracy within the train data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3,random_state=42)


#independent variable for test data
hr_test = hr_test.drop(['employee_id'],1)




#scale down for train
scale = StandardScaler()
X_train_scale = scale.fit_transform(X)
X = pd.DataFrame(X_train_scale)

#scale down for test
X_test_scale = scale.fit_transform(hr_test)
hr_test = pd.DataFrame(X_test_scale)

###############Alogorithm

#logistic
classifier = LogisticRegression(random_state=1)
classifier.fit(X, Y)
y_pred = classifier.predict(X_test)
y_pred=pd.DataFrame(y_pred) 

y_out=classifier.predict(hr_test)
y_out.to_csv("C:/Users/ssn/Desktop/capstone/abalytics/LOG.csv")





#decision

classifier= DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(X_train,y_train) 
y_pred = classifier.predict(X_test)

y_out=classifier.predict(hr_test)
y_out.to_csv("C:/Users/ssn/Desktop/capstone/abalytics/DEC.csv")

#naive
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_out=classifier.predict(hr_test)
y_out.to_csv("C:/Users/ssn/Desktop/capstone/abalytics/naive.csv")



svc=SVC()

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)


y_pred = svc.predict(hr_test)

y_pred=pd.DataFrame(y_pred) 

y_pred.to_csv("C:/Users/ssn/Desktop/capstone/abalytics/svc.csv")


###Adaboost using decision tree



classifier= DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
 max_features=None, max_leaf_nodes=None,
 min_impurity_split=1e-07, min_samples_leaf=1,
 min_samples_split=2, min_weight_fraction_leaf=0.0,
 presort=False, random_state=None, splitter='best')

# Create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=50, base_estimator=classifier,learning_rate=1)

abc.fit(X_train, y_train)

y_pred = abc.predict(X_test)

y_pred = abc.predict(hr_test)


y_pred=pd.DataFrame(y_pred) 

y_pred.to_csv("C:/Users/ssn/Desktop/capstone/abalytics/abc1.csv")

#Gradient boost algorithm
#####higeht
modelXg = XGBClassifier(silent=False, 
                      scale_pos_weight=3,
                      learning_rate=0.07,  
                      colsample_bytree = 0.9,
                      min_child_weight=4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=400, 
                      reg_alpha = 0.3,
                      nthread=9,
                      max_depth=3, 
                      gamma=0.4)
#####higeht
modelXg = XGBClassifier(silent=False, 
                      scale_pos_weight=3,
                      learning_rate=0.06,  
                      colsample_bytree = 0.9,
                      min_child_weight=4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=700, 
                      reg_alpha = 0.3,
                      nthread=11,
                      max_depth=3, 
                      gamma=0.4)

modelXg.fit(X_train, y_train)
y_pred = modelXg.predict(X_test)


y_xg = modelXg.predict(hr_test)


y_xg=pd.DataFrame(y_xg) 

y_xg.to_csv("C:/Users/ssn/Desktop/capstone/abalytics/GB8.csv")


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


#overall GB algorithm gives better accuracy
all=precision_recall_fscore_support(y_test, y_pred, average='macro')
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)