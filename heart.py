import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


# Loading dataset and checking null and statistical summaries


df=pd.read_csv("C:\\Users\\HP\\OneDrive\\Documents\\Project(Data Science)\\Heart Disease\\heart.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Using Label encoder to convert categorical data into numeric for better model training
# Categorical data: Sex, ChestPainType, RestingECG, ExerciseAgina, ST_Slope

sex_encoder=LabelEncoder()
chestPain_encoder=LabelEncoder()
restingECG_encodeder=LabelEncoder()
ExerciseAgina_encoder=LabelEncoder()
st_slopeencoder=LabelEncoder()

df['Sex']=sex_encoder.fit_transform(df['Sex'])
df['ChestPainType']=chestPain_encoder.fit_transform(df['ChestPainType'])
df['RestingECG']=restingECG_encodeder.fit_transform(df['RestingECG'])
df['ExerciseAngina']=ExerciseAgina_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope']=st_slopeencoder.fit_transform(df['ST_Slope'])


# Using HeatMap to find correlation amoung all column and target column
sns.heatmap(df.corr(),annot=True)
plt.show()

# print(df.columns)


# Using VIF to find multicollinearity and remove columns with high VIF
# x is the dataset without 'HeartDisease' as it is the target value

# x=df.drop(['HeartDisease'],axis=1) 
# vif_data=pd.DataFrame()
# vif_data['Feature']=x.columns
# vif_data['VIF']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
# print(vif_data)


# Dropping all those column with higher vif [> 15] and rechecking VIF
# ['RestingBP','HeartDisease'] have higher vif

x=df.drop(['HeartDisease','RestingBP','MaxHR'],axis=1)
vif_data=pd.DataFrame()
vif_data['Feature']=x.columns
vif_data['VIF']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
print(vif_data)


# Training model using Logistic Regression

y=df['HeartDisease']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)

model=LogisticRegression(max_iter=1000)  #max_iter tells the maximum no. of iteration the solver is allowed to perform, to find the best optimal solution
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# Evaluation 
print("classification report : ",classification_report(y_test,y_pred))
print("\n")
print("confusion matrix: ",confusion_matrix(y_test,y_pred))
print("\n")
print("accuracy score: ",accuracy_score(y_test,y_pred))  # model got 84.23% of test smaple right


# Saving the trained models
with open('heart_disease.pkl','wb')as f:
    pickle.dump(model,f)

with open('sex.pkl','wb')as f:
    pickle.dump(sex_encoder,f)

with open('chestPainType.pkl','wb')as f:
    pickle.dump(chestPain_encoder,f)

with open('restingECG.pkl','wb')as f:
    pickle.dump(restingECG_encodeder,f)

with open('exerciseAngina.pkl','wb')as f:
    pickle.dump(ExerciseAgina_encoder,f)

with open('stSlope.pkl','wb')as f:
    pickle.dump(st_slopeencoder,f)


