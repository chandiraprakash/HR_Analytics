import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


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


#filling the missing values
hr_df['education'][hr_df['region'] == 'region_10']=hr_df['education'][hr_df['region'] == 'region_10'].fillna(hr_df['education'][hr_df['region'] == 'region_10'].mode()[0])
hr_df['education'][hr_df['region'] != 'region_10']=hr_df['education'][hr_df['region'] != 'region_10'].fillna(hr_df['education'][hr_df['region'] != 'region_10'].mode()[0])

hr_df['previous_year_rating']=hr_df['previous_year_rating'].fillna(hr_df['previous_year_rating'].mode()[0])



#filling the missing values for test data
hr_test['education'][hr_test['region'] == 'region_10']=hr_test['education'][hr_test['region'] == 'region_10'].fillna(hr_test['education'][hr_test['region'] == 'region_10'].mode()[0])
hr_test['education'][hr_test['region'] != 'region_10']=hr_test['education'][hr_test['region'] != 'region_10'].fillna(hr_test['education'][hr_test['region'] != 'region_10'].mode()[0])

hr_test['previous_year_rating']=hr_test['previous_year_rating'].fillna(hr_test['previous_year_rating'].mode()[0])

#dropping the columns
del hr_df['age']
del hr_df['region']
#dropping the columns in test
del hr_test['age']
del hr_test['region']


#converting categorical data into numbers

hr_df=pd.get_dummies(hr_df,columns=['department','education','gender','recruitment_channel'])

#converting categorical data into numbers in test data
hr_test=pd.get_dummies(hr_test,columns=['department','education','gender','recruitment_channel'])


#to find the list of columns
list(hr_df.columns.values)

list(hr_test.columns.values)

list(x_hr.columns.values)


#splitting the dep and indep of the train data
X = hr_df.iloc[:,[1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
Y = hr_df.iloc[:,7]

#to find the accuracy within the train data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3,random_state=42,shuffle=True)


#independent variable for test data
x_hr=hr_test.iloc[:,1:24]

x_hr.head(1)
#After predicting find the accuracy
X_test=hr_test.iloc[:,[1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
Y_test=hr_test.iloc[:,7]

#scale down for train
scale = StandardScaler()
X_train_scale = scale.fit_transform(X)
X = pd.DataFrame(X_train_scale)

#scale down for test
X_test_scale = scale.fit_transform(x_hr)
x_hr = pd.DataFrame(X_test_scale)

#logistic
classifier = LogisticRegression(random_state=1)
classifier.fit(X, Y)


#predict the Y values
y_pred = classifier.predict(x_hr)

y_pred=pd.DataFrame(y_pred) 

y_pred.to_csv("C:/Users/ssn/Desktop/capstone/Y_pred1.csv")

confusion_matrix = confusion_matrix(Y_test, y_pred)
print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

all=precision_recall_fscore_support(Y_test, y_pred, average='macro')
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)
