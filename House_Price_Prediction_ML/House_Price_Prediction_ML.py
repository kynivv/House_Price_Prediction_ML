import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_excel("HousePricePrediction.xlsx")

### Dataset Preprocessing
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
#print('Categorical Vars: ', len(object_cols))

int_ = (dataset.dtypes == 'int64')
num_cols = list(int_[int_].index)
#print('Int Vars: ', len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
#print('Float Vars: ', len(fl_cols))

dataset_num_vars = dataset.select_dtypes(include= ('int64', 'float')) #Dataset's numerical colums

plt.figure(figsize=(12, 6))
sns.heatmap(dataset_num_vars.corr(),cmap='BrBG', fmt='.2f', linewidths=2, annot=True )
#plt.show()


# Unique Values
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('Unique values of Categorical Features')

sns.barplot(x=object_cols, y=unique_values)
#plt.show()


# Unique Values per each category
plt.figure(figsize=(18, 36))
plt.title('Categorical Featres: Distribution')
plt.xticks(rotation=90)

index = 1
for col in object_cols :
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation = 90)
    sns.barplot(x=list(y.index), y=y)
    index += 1

#plt.show()


# Data Cleaning
dataset.drop(['Id'], axis=1, inplace= True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
new_dataset = dataset.dropna()

#print(new_dataset.isnull().sum())


# OneHotEncoder categorical data into binary
from sklearn.preprocessing import OneHotEncoder

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
#print('Categorical vars: ', object_cols)
#print('No. of categorical features: ', len(object_cols))

OH_encoder = OneHotEncoder(sparse= False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)



# Splitting Dataset into Training and Testing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)


### Training Model

#SVM
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)

#accuracy
print('Accuracy: ', round((100 * (1-(mean_absolute_percentage_error(Y_valid, Y_pred)))), 2), "%")
