#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import basic libraries 
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import plot libraries 
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# # BIG MART SALES DATASET

# In[2]:


from IPython.display import Image
path = "C:\\Users\\Thanh\\Documents\\Data mining\\Data_Mining_Lab04\\"
Image(filename= path + "BigMartSales01.jpg")


# In[3]:


Image(filename= path + "BigMartSales02.png") 


# In[4]:


# Read data
data = pd.read_csv(path + "BigMartSales.csv")
data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# - Cột "Item_Weight" có ít giá trị (count) hơn các cột còn lại, 7060 so với 8523. Việc này chứng tỏ có 1463 giá trị bị khuyết ở đây

# In[7]:


#Check for duplicates
idsUnique = len(set(data.Item_Identifier))
idsTotal = data.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")


# In[8]:


data.hist(bins=50, figsize=(20, 15))


# In[9]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(data.Item_Outlet_Sales, bins = 25)
#plt.ticklabel_format(style='plain', axis='x', scilimits=(0,1))
plt.xlabel("Item_Outlet_Sales")
plt.ylabel("Number of Sales")
plt.title("Item_Outlet_Sales Distribution")


# In[10]:


print ("Skew is:", data.Item_Outlet_Sales.skew())
print("Kurtosis: %f" % data.Item_Outlet_Sales.kurt())


# In[11]:


numeric_features = data.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[12]:


corr =numeric_features.corr()
corr


# In[13]:


#correlation matrix
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True)


# #### Item_Fat_Content

# In[14]:


sns.countplot(data.Item_Fat_Content)


# For Item_Fat_Content there are two possible choices : “Low Fat” or “Regular”. However, in our data we have these two types of Fat writen in different manners. This must be corrected

# #### Item_Type

# In[15]:


sns.countplot(data.Item_Type)
plt.xticks(rotation=90)


# Looking at the list of Item_Type we see there are sixteen different types. This is a high number of unique values for a categorical variable. Therefore we must try to think of a way to drastically reduce this number

# #### Outlet_Size

# In[16]:


sns.countplot(data.Outlet_Size)


# There seems to be a low number of stores with size equals to “High”. Most of the existent stores seem to be either “Small” or “Medium”. It will be interesting to see how this variable relates to our target. If “High” size stores have better results as initially expected or due to this number distribution sales results might be similar.

# #### Outlet_Location_Type 

# In[17]:


sns.countplot(data.Outlet_Location_Type)


# Bigmart appears to be a supermarket brand that is more present in “Small” to “Medium” size cities than in more densily populated locations.
# 
# 

# #### Outlet_Type 

# In[18]:


sns.countplot(data.Outlet_Type)
plt.xticks(rotation=90)


# It looks like Supermarket Type2 , Grocery Store and Supermarket Type3 all have low expression in this distribution. Maybe we can create a single category with all of the three. Nevertheless, before doing this we must see their impact in the Item_Outlet_Sales .
# 
# Firstly we individually analysed some of the existent features, now it is time to understand the relationship between our target variable and predictors as well as the relationship among predictors

# In[19]:


plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
plt.plot(data.Item_Weight, data["Item_Outlet_Sales"],'.', alpha = 0.3)


# We saw previously that Item_Weight had a low correlation with our target variable. If we plot both features we can see that relationship

# #### Item_Weight 

# In[20]:


plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
plt.plot(data.Item_Visibility, data["Item_Outlet_Sales"],'.', alpha = 0.3)


# Visibility in Store: The location of product in a store will impact sales. Ones which are right at entrance will catch the eye of customer first rather than the ones in back.
# 
# This was the assumption made… however, first the correlation and now this plot chart, indicate that the more visible a product is the less higher its sales will be. This might be due to the fact that a great number of daily use products, which do not need high visibility, control the top of the sales chart. As we can see from the bar charts below, most sold products have lower visibility. Furthermore, there is a concerning number of products with visibility zero.

# #### Outlet_Establishment_Year và Item_Outlet_Sales

# In[21]:


Outlet_Establishment_Year_pivot = data.pivot_table(index='Outlet_Establishment_Year', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# There seems to be no significant meaning between the year of store establishment and the sales for the items. 1998 has low values but thet might be due to the fact the few stores opened in that year.

# ####  Item_Fat_Content và Item_Outlet_Sales

# In[22]:


Item_Fat_Content_pivot = data.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# Daily use products should have a higher tendency to sell as compared to the specific use products. “Low Fat” products seem to have higher sales values than “Regular” products.

# #### Outlet_Identifier và Item_Outlet_Sales

# In[23]:


Outlet_Identifier_pivot = data.pivot_table(index='Outlet_Identifier', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Identifier")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Identifier on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# Cửa hàng có mã ID là OUT027 có doanh số bán hàng cao nhất và OUT010 thấp nhất, tiếp theo là đến cửa hàng OUT019.

# In[24]:


data.pivot_table(values='Outlet_Type', columns='Outlet_Identifier',aggfunc=lambda x:x.mode())


# In[25]:


data.pivot_table(values='Outlet_Type', columns='Outlet_Size',aggfunc=lambda x:x.mode())


# From the ten stores, two are Groceries whereas six are Supermarket Type1, one Supermarket Type2 and one Supermarket Type3. You can get this information from the pivot_tables below.
# 
# From the above bar chart, we see that thr groceries (“OUT010”, “OUT019”) have the lowest sales results which is expected followed by the Supermarket Type 2 (“OUT018”). Curiously, most stores are of type Supermarket Type1 of size “High” and do not have the best results. The best results belong to “Out027” which is a “Medium” size Supermarket Type 3.

# ####  Outlet_Size và Item_Outlet_Sales

# In[26]:


Outlet_Size_pivot = data.pivot_table(index='Outlet_Size', values='Item_Outlet_Sales', aggfunc=np.median)
Outlet_Size_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Size")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Size on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In the beginning, our belief was that stores which are very big in size should have higher sales as they act like one-stop-shops and people would prefer getting everything from one place. According to the results, this is almost the case. Curiously, consumers tend to prefer medium size stores instead of big size. As we saw in the previous section, most stores have size “Medium” but still the “High” and “Small” stores which are clearly in an inferior number can beat or even come close to their numbers

# #### Outlet_Type và Item_Outlet_Sales

# In[27]:


Outlet_Type_pivot = data.pivot_table(index='Outlet_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Type ")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# Supermarket Type3 thì có doanh thhu cao nhất và cửa hàng tạp hóa có doanh thu thấp nhất.

# ####  Outlet_Location_Type và Item_Outlet_Sales

# In[28]:


Outlet_Location_Type_pivot = data.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Location_Type_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Location_Type ")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Location_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[29]:


data.pivot_table(values='Outlet_Location_Type', columns='Outlet_Type',aggfunc=lambda x:x.mode())


# Do Tier 1 cities have higher sales? This was one of the premisses we made in the start of this study. However, if we look at our results we see that in fact it is stores from Tier 2 cities that present the highest results, followed by Tier 3 cities and with Tier 1 cities with the lowest results of the three type of locations.
# 
# From the pivot_table it is easy to see that Tier2 and Tier3 cities are those that have highest representation of stores.
# 
# During our EDA we were able to take some conclusions regarding our first assumptions and the available data:
# 
# Regarding the variables which were thought to have high impact on the product’s sale price.
# 
# Item_Visibility does not have a high positive correlation as expected, quite the opposite. As well, there are no big variations in the sales due to theItem_Type . On the other hand, it was possible to see that the size, location and type of store could have a positive impact on sales.
# 
# If we look at variable Item_Identifer , we can see different groups of letters per each product such as ‘FD’ (Food), ‘DR’(Drinks) and ‘NC’ (Non-Consumable). From this we can create a new variable.
# 
# Regarding Item_Visibility there are items with the value zero. This does not make lot of sense, since this is indicating those items are not visible on the store.
# 
# Similar, Item_Weight and Outlet_Size seem to present NaN values.
# 
# There seems to be 1562 unique items only available in a single store.
# 
# Item_Fat_Content has vale “low fat” writen in different manners.
# 
# For Item_Type we try to create a new feature that does not have 16 unique values.
# 
# Outlet_Establishment_Year besides being a hidden category, its values vary from 1985 to 2009 . It must be converted to how old the store is to better see the impact on sales.

# In[30]:


#Check the percentage of null values per variable
data.isnull().sum()/data.shape[0]*100 #show values in percentage


# In[31]:


#aggfunc is mean by default! Ignores NaN by default
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
print(item_avg_weight.head())


# In[32]:


data["Outlet_Size"] = data["Outlet_Size"].fillna(data["Outlet_Size"].mode()[0])
data["Item_Weight"] = data["Item_Weight"].fillna(np.nanmedian(data["Item_Weight"]))
#Check the percentage of null values per variable
data.isnull().sum()/data.shape[0]*100 #show values in percentage


# In[33]:


#Creates pivot table with Outlet_Type and the mean of #Item_Outlet_Sales. Agg function is by default mean()
data.pivot_table(values='Item_Outlet_Sales', columns='Outlet_Type')


# Liệu việc kết hợp Siêu thị Loại 2 và Loại 3 có khả thi không? Hãy kiểm tra.
# 
# Theo quan sát của kết quả bên trên, doanh số bán sản phẩm trung bình khác nhau đáng kể, vì vậy chúng tôi để nguyên

# In[34]:


# thay giá trị 0 bằng giá trị trung bình trên 'Item_Visibility'
print ('Original #zeros: %d'%sum(data['Item_Visibility'] == 0))
data['Item_Visibility']=data['Item_Visibility'].replace(0,data['Item_Visibility'].mean())
print ('Final #zeros: %d'%sum(data['Item_Visibility'] == 0))


# In[35]:


#Remember the data is from 2013
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# We talked about using how long has been working instead of the date of start. Remember that the data we have is from 2013. Thus we must consider this year into our calculations

# In[36]:


#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[37]:


#Change categories of low fat:
print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())
print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())


# In[38]:


#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


# In[39]:


#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[40]:


#Dummy Variables:
data = pd.get_dummies(data, columns =['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
data.dtypes


# In[41]:


data.shape


# In[42]:


# Split Train and Test and check shape 
def SplitTrainAndTest(data, TrainRate, TargetAtt):
    # gets a random TrainDataRate % of the entire set
    training = data.sample(frac=TrainRate, random_state=1)
    # gets the left out portion of the dataset
    testing = data.loc[~data.index.isin(training.index)]

    data_train = training.drop(TargetAtt, 1)
    label_train = training[[TargetAtt]]
    data_test = testing.drop(TargetAtt, 1)
    label_test = testing[[TargetAtt]]

    PrintTrainTestInfo(data_train, label_train, data_test, label_test)
    return data_train, label_train, data_test, label_test
    
def PrintTrainTestInfo(data_train, label_train, data_test, label_test):
  print("Train shape : ", data_train.shape)
  print("Test shape : ", data_test.shape)


# In[43]:


from sklearn.preprocessing import LabelEncoder
EncoderAttList = ["Item_Type"]

data_encoder = data.copy()
for att in EncoderAttList:
  data_encoder[att] = LabelEncoder().fit_transform(data_encoder[att])

data_encoder = data_encoder.drop(['Item_Identifier', 'Outlet_Identifier'], 1)  

display(data_encoder.info())
display(data_encoder.head(10))


# In[44]:


data_train, label_train, data_test, label_test = SplitTrainAndTest(data_encoder, 0.7, 'Item_Outlet_Sales')
FeatureList = data_train.columns.to_list()
TargetAtt = 'Item_Outlet_Sales'
print(FeatureList)


# In[45]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

LRModel = LinearRegression(normalize=True)
Model = LRModel

Model.fit(data_train[FeatureList], label_train)
# model evaluation for training set
label_train_predict = Model.predict(data_train[FeatureList])
rmse = (np.sqrt(mean_squared_error(label_train, label_train_predict)))
r2 = r2_score(label_train, label_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
label_test_predict = Model.predict(data_test[FeatureList])
rmse = (np.sqrt(mean_squared_error(label_test, label_test_predict)))
r2 = r2_score(label_test, label_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[46]:


PredictDF = data_test[FeatureList].copy()
PredictDF[TargetAtt] = label_test
PredictDF["Predict"] = label_test_predict
PredictDF = PredictDF.reset_index(drop=False)

display(PredictDF.head())


# In[47]:


from sklearn.tree import DecisionTreeRegressor
DTModel = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
Model = DTModel

Model.fit(data_train[FeatureList], label_train)
# model evaluation for training set
label_train_predict = Model.predict(data_train[FeatureList])
rmse = (np.sqrt(mean_squared_error(label_train, label_train_predict)))
r2 = r2_score(label_train, label_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
label_test_predict = Model.predict(data_test[FeatureList])
rmse = (np.sqrt(mean_squared_error(label_test, label_test_predict)))
r2 = r2_score(label_test, label_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[48]:


from sklearn.ensemble import RandomForestRegressor
RFModel = RandomForestRegressor(n_estimators=100,max_depth=8, min_samples_leaf=150, random_state=123)
Model = RFModel

Model.fit(data_train[FeatureList], label_train)
# model evaluation for training set
label_train_predict = Model.predict(data_train[FeatureList])
rmse = (np.sqrt(mean_squared_error(label_train, label_train_predict)))
r2 = r2_score(label_train, label_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
label_test_predict = Model.predict(data_test[FeatureList])
rmse = (np.sqrt(mean_squared_error(label_test, label_test_predict)))
r2 = r2_score(label_test, label_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[49]:


def XGBRegressorApporach(data_train, label_train, data_test, label_test, FeatureList, TargetAtt):
  from sklearn.metrics import mean_squared_error
  from sklearn.metrics import r2_score
  from xgboost import XGBRegressor

  XGBModel = XGBRegressor()
  XGBModel.fit(data_train[FeatureList], label_train)

  # model evaluation for training set
  label_train_predict = XGBModel.predict(data_train[FeatureList])
  rmse = (np.sqrt(mean_squared_error(label_train, label_train_predict)))
  r2 = r2_score(label_train, label_train_predict)

  print("The model performance for training set")
  print("--------------------------------------")
  print('RMSE is {}'.format(rmse))
  print('R2 score is {}'.format(r2))
  print("\n")

  # model evaluation for testing set
  label_test_predict = XGBModel.predict(data_test[FeatureList])
  rmse = (np.sqrt(mean_squared_error(label_test, label_test_predict)))
  r2 = r2_score(label_test, label_test_predict)

  print("The model performance for testing set")
  print("--------------------------------------")
  print('RMSE is {}'.format(rmse))
  print('R2 score is {}'.format(r2))
  print("\n")

  PredictDF = data_test[FeatureList].copy()
  PredictDF[TargetAtt] = label_test
  PredictDF["Predict"] = label_test_predict
  PredictDF = PredictDF.reset_index(drop=False)

  print("The the predict table result : ")
  print("--------------------------------------")
  display(PredictDF.head(10))
  
  return XGBModel, PredictDF

def PredictXGBRegressor(XGBModel, FeatureSample, TrueLabel = ""):
  if isinstance(FeatureSample, pd.Series):
    FeatureSample = [FeatureSample]
  
  print(FeatureSample)

  label_predict = XGBModel.predict(FeatureSample)
  
  
  DirectPredictDF = FeatureSample.copy()
  DirectPredictDF["LabelTest"] = list(TrueLabel)
  DirectPredictDF["Predict"] = label_predict
  DirectPredictDF = DirectPredictDF.reset_index(drop=False)
  
  print("\n")
  print("The the predict table result : ")
  print("--------------------------------------")
  display(DirectPredictDF.head(10))
  
  return DirectPredictDF


# In[50]:


XGBModel, PredictDF = XGBRegressorApporach(data_train, label_train, data_test, label_test, FeatureList, TargetAtt)


# In[51]:


def LGBMRegressorApporach(data_train, label_train, data_test, label_test, FeatureList, TargetAtt):
  from sklearn.metrics import mean_squared_error
  from sklearn.metrics import r2_score
  import lightgbm as lgb

  LGBMModel = lgb.LGBMRegressor()
  LGBMModel.fit(data_train[FeatureList], label_train)

  # model evaluation for training set
  label_train_predict = LGBMModel.predict(data_train[FeatureList])
  rmse = (np.sqrt(mean_squared_error(label_train, label_train_predict)))
  r2 = r2_score(label_train, label_train_predict)

  print("The model performance for training set")
  print("--------------------------------------")
  print('RMSE is {}'.format(rmse))
  print('R2 score is {}'.format(r2))
  print("\n")

  # model evaluation for testing set
  label_test_predict = LGBMModel.predict(data_test[FeatureList])
  rmse = (np.sqrt(mean_squared_error(label_test, label_test_predict)))
  r2 = r2_score(label_test, label_test_predict)

  print("The model performance for testing set")
  print("--------------------------------------")
  print('RMSE is {}'.format(rmse))
  print('R2 score is {}'.format(r2))
  print("\n")

  PredictDF = data_test[FeatureList].copy()
  PredictDF[TargetAtt] = label_test
  PredictDF["Predict"] = label_test_predict
  PredictDF = PredictDF.reset_index(drop=False)

  print("The the predict table result : ")
  print("--------------------------------------")
  display(PredictDF.head(10))
  
  return LGBMModel, PredictDF

def PredictLGBMRegressor(LGBMModel, FeatureSample, TrueLabel = ""):
  if isinstance(FeatureSample, pd.Series):
    FeatureSample = [FeatureSample]
  
  print(FeatureSample)

  label_predict = LGBMModel.predict(FeatureSample)
  
  
  DirectPredictDF = FeatureSample.copy()
  DirectPredictDF["LabelTest"] = list(TrueLabel)
  DirectPredictDF["Predict"] = label_predict
  DirectPredictDF = DirectPredictDF.reset_index(drop=False)
  
  print("\n")
  print("The the predict table result : ")
  print("--------------------------------------")
  display(DirectPredictDF.head(10))
  
  return DirectPredictDF


# In[52]:


LGBMModel, PredictDF = LGBMRegressorApporach(data_train, label_train, data_test, label_test, FeatureList, TargetAtt)


# # CUSTOMER CHURN DATASET

# In[53]:


path = "C:\\Users\\Thanh\\Documents\\Data mining\\Data_Mining_Lab04\\"
Image(filename= path + "CustomerChurn.png")


# In[54]:


# Read data
data = pd.read_csv('C:\\Users\\Thanh\\Documents\\Data mining\\Data_Mining_Lab04\\CustomerChurn.csv')
data.head()


# In[55]:


data.info()


# In[56]:


print(" List of unique values in State : ")
print(data['State'].unique())
print(" List of unique values in International plan : ")
print(data['International plan'].unique())
print(" List of unique values in Voice mail plan : ")
print(data['Voice mail plan'].unique())

#Special Field
print(" List of unique values in Area code : ")
print(data['Area code'].unique())


# In[57]:


data.describe()


# In[58]:


dataNChurn = data[data['Churn'] == False]
dataNChurn.head()


# In[59]:


dataNChurn.describe()


# #### Total day minutes và Total day calls

# In[60]:


sns.set(color_codes=True)
sns.distplot(data['Total day minutes'], bins=20)
df = pd.DataFrame(dataNChurn, columns=['Total day minutes', 'Total day calls'])
df = df.reset_index(drop=True)
sns.jointplot(x='Total day minutes', y='Total day calls', data=df)


# In[61]:


data.describe(include=['O'])


# #### State

# In[62]:


Distribution = data['State'].value_counts()
Distribution = pd.DataFrame({'State':Distribution.index, 'Freq':Distribution.values})
Distribution = Distribution.sort_values(by='State', ascending=True)
plt.subplots(figsize=(18,5))
plt.bar(Distribution['State'], Distribution["Freq"])
plt.xticks(Distribution['State'])
plt.ylabel('Frequency')
plt.title('Barplot of ' + 'State')
plt.show()


# Các Customer của chúng ta phân bố đều các State chỉ có State WV là nhiều bất thường nhất

# In[63]:


def DrawBoxplot2(DataFrame, xAtt, yAtt, hAtt="N/A"):
  plt.figure()
  plt.subplots(figsize=(10,5))
  if(hAtt == "N/A"):
    sns.boxplot(x=xAtt, y=yAtt,  data=DataFrame)
  else:
    sns.boxplot(x=xAtt, y=yAtt,  hue=hAtt,  data=DataFrame)
  plt.show()


# ####  Total day minutes

# In[64]:


DrawBoxplot2(data, xAtt = 'Churn', yAtt='Total day minutes')
DrawBoxplot2(data, xAtt = 'Churn', yAtt='Total day minutes', hAtt = 'International plan')


# In[65]:


def DrawCountplot(DataFrame, att, hatt="N/A"):
  if(hatt == "N/A"):
    sns.countplot(x=att, data=DataFrame)
  else:
    sns.countplot(x=att, hue=hatt, data=DataFrame)
  plt.show()
  
def DrawHistogram(DataFrame, att):
  import matplotlib.pyplot as plt
  plt.figure()
  DataFrame[att].hist(edgecolor='black', bins=20)
  plt.title(att)
  plt.show()


# ####  Area code theo Churn

# In[66]:


DrawCountplot(data, 'Area code', 'Churn')


# ####  Customer service calls theo Churn

# In[67]:


DrawCountplot(data, 'Customer service calls', 'Churn')


# ####  Account length

# In[68]:


DrawHistogram(data,'Account length')


# In[69]:


from sklearn.preprocessing import LabelEncoder
data_encoder = data.copy()
data_encoder['State'] = LabelEncoder().fit_transform(data_encoder['State'])
data_encoder['International plan'] = LabelEncoder().fit_transform(data_encoder['International plan'])
data_encoder['Voice mail plan'] = LabelEncoder().fit_transform(data_encoder['Voice mail plan'])
data_encoder = data_encoder.join(pd.get_dummies(data_encoder['Area code'], prefix='Area_code_'))
data_encoder = data_encoder.drop('Area code', axis=1)
data_encoder.head(10)


# In[70]:


ColumnList = data_encoder.columns
ColumnList = list(set(ColumnList) - set(['Churn'])) 
ColumnList.append('Churn') 
print(ColumnList)


# In[71]:


data_encoder = data_encoder[ColumnList]
display(data_encoder.head(10))


# In[72]:


# Split Train and Test and check shape 
def SplitTrainAndTest(data, TrainRate, TargetAtt):
    # gets a random TrainDataRate % of the entire set
    training = data.sample(frac=TrainRate, random_state=1)
    # gets the left out portion of the dataset
    testing = data.loc[~data.index.isin(training.index)]

    data_train = training.drop(TargetAtt, 1)
    label_train = training[[TargetAtt]]
    data_test = testing.drop(TargetAtt, 1)
    label_test = testing[[TargetAtt]]

    PrintTrainTestInfo(data_train, label_train, data_test, label_test)
    return data_train, label_train, data_test, label_test
    
def PrintTrainTestInfo(data_train, label_train, data_test, label_test):
  print("Train shape : ", data_train.shape)
  print("Test shape : ", data_test.shape)


# In[73]:


data_train, label_train, data_test, label_test = SplitTrainAndTest(data_encoder, TrainRate = 0.7, TargetAtt = 'Churn')


# In[74]:


def NaiveBayesLearning(DataTrain, TargetTrain):
    from sklearn.naive_bayes import GaussianNB
    NBModel = GaussianNB()
    NBModel.fit(DataTrain, TargetTrain.values.ravel())

    return NBModel

def NaiveBayesTesting(NBModel,DataTest, TargetTest):
    from sklearn.metrics import accuracy_score
    PredictTest = NBModel.predict(DataTest)
    Accuracy = accuracy_score(TargetTest, PredictTest)

    return Accuracy, PredictTest


# In[75]:


def LogisticRegressionLearning(DataTrain, TargetTrain):
    # Apply the Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    logreg = LogisticRegression()
    # training by Logistic Regression
    logreg.fit(DataTrain, TargetTrain.values.ravel())

    return logreg

def LogisticRegressionTesting(LRModel,DataTest, TargetTest):
    # Testing and calculate the accuracy
    from sklearn.metrics import accuracy_score

    logreg = LRModel
    PredictTest = logreg.predict(DataTest)
    Accuracy = accuracy_score(TargetTest, PredictTest)
    # print('Logistic regression accuracy: {:.3f}'.format(Accuracy))

    return Accuracy, PredictTest


# In[76]:


def RandomForestLearning(DataTrain, TargetTrain):
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators = 500)
    rf.fit(DataTrain, TargetTrain.values.ravel())

    return rf

def RandomForestTesting(RFModel,DataTest, TargetTest):
    from sklearn.metrics import accuracy_score

    PredictTest = RFModel.predict(DataTest)
    Accuracy = accuracy_score(TargetTest, PredictTest)
    # print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(TargetTest, PredictTest)))

    return Accuracy, PredictTest


# In[77]:


def SVMLearning(DataTrain, TargetTrain, ClassifierType = " "):
    from sklearn.svm import SVC
    if(ClassifierType == 'Linear'):
        svc = SVC(kernel="linear", C=0.025)
#         print('SVM Linear processing')
    # Radial basis function kernel
    elif (ClassifierType == 'RBF'):
        svc = SVC(gamma=2, C=1)
#         print('SVM RBF processing')
    else:
        svc = SVC()
#         print('SVM Default processing')
    svc.fit(DataTrain, TargetTrain.values.ravel())
    return svc

def SVMTesting(SVMModel, DataTest, TargetTest):
    from sklearn.metrics import accuracy_score
    PredictTest = SVMModel.predict(DataTest)
    Accuracy = accuracy_score(TargetTest, PredictTest)
    # print('Support Vector Machine Accuracy: {:.3f}'.format(accuracy_score(TargetTest, PredictTest)))
    return Accuracy, PredictTest


# In[78]:


X_train = data_train
y_train = label_train
X_test = data_test
y_test = label_test


# In[79]:


NBModel = NaiveBayesLearning(X_train, y_train)
NBAccuracy,NBPredictTest = NaiveBayesTesting(NBModel,X_test, y_test)
print('Naive Bayes accuracy: {:.3f}'.format(NBAccuracy))

print(NBModel.predict_proba(X_test)[1,:])
print(NBPredictTest[1])
print(NBModel.predict_proba(X_test)[0,:])
print(NBPredictTest[0])


# In[80]:


LRModel = LogisticRegressionLearning(X_train, y_train)
LRAccuracy,LRPredictTest = LogisticRegressionTesting(LRModel,X_test, y_test)
print('Logistic Regression accuracy: {:.3f}'.format(LRAccuracy))

print(LRModel.predict_proba(X_test)[1,:])
print(LRPredictTest[1])
print(LRModel.predict_proba(X_test)[0,:])
print(LRPredictTest[0])


# In[81]:


RFModel = RandomForestLearning(X_train, y_train)
RFAccuracy,RFPredictTest = RandomForestTesting(RFModel,X_test, y_test)
print('Random Forest accuracy: {:.6f}'.format(RFAccuracy))

print(RFModel.predict_proba(X_test)[1,:])
print(RFPredictTest[1])
print(RFModel.predict_proba(X_test)[0,:])
print(RFPredictTest[0])


# In[82]:


LiSVMModel = SVMLearning(X_train, y_train)
LiSVMAccuracy,LiSVMPredictTest = SVMTesting(LiSVMModel, X_test, y_test)
print('Linear SVM accuracy: {:.6f}'.format(LiSVMAccuracy))

RBFSVMModel = SVMLearning(X_train, y_train, 'RBF')
RBFSVMAccuracy,RBFSVMPredictTest = SVMTesting(RBFSVMModel, X_test, y_test)
print('RBF SVM accuracy: {:.6f}'.format(RBFSVMAccuracy))


# In[83]:


sns.pairplot(data, vars=["Total day minutes", "Total day calls", "Total day charge"], hue="Churn")


# In[84]:


def DrawCorrelationMap(data, AttList):
  import seaborn as sns 
  correlation_matrix = data.loc[:,AttList].corr().round(2)
  print("Correlation Matrix : ")
  print(correlation_matrix.to_string())
  # annot = True to print the values inside the square
  sns.heatmap(data=correlation_matrix, cmap = 'coolwarm', annot=True)
  
  return correlation_matrix


# In[85]:


AttList = [ "Total day minutes", "Total day calls","Total day charge","Total night minutes","Total night calls","Total night charge"]
correlation_matrix = DrawCorrelationMap(data, AttList)


# In[86]:


def BayesianMLApproach(data_train, label_train, data_test, label_test, FeatureList, TargetAtt, TargetNames):
  import warnings
  warnings.filterwarnings("ignore")

  # Instantiate the classifier
  from sklearn.naive_bayes import GaussianNB
  BayesModel = GaussianNB()
  
  # Train classifier
  BayesModel.fit( data_train[FeatureList].values, label_train)
  y_pred = BayesModel.predict(data_test[FeatureList])

  # Print results
  RowsNum = data_test.shape[0]
  mislabeledNum =  (label_test[TargetAtt] != y_pred).sum()
  correctlabeledNum = RowsNum - mislabeledNum
  print("Number of correct labeled points out of a total {} points : {}"
        .format(RowsNum, correctlabeledNum))
  print("Number of mislabeled points out of a total {} points : {}"
        .format(RowsNum,mislabeledNum))
  print('Accuracy of classifier on training set: {:.2f}'
     .format(BayesModel.score(data_train[FeatureList], label_train)))
  print('Accuracy of classifier on test set: {:.2f}'
     .format(BayesModel.score(data_test[FeatureList], label_test)))
  
  ColumnNames = ["Proba_" + s for s in TargetNames]
  PredictDF = pd.DataFrame(BayesModel.predict_proba(data_test[FeatureList]), columns = ColumnNames)
  PredictDF["LabelTest"] = list(label_test[TargetAtt])
  PredictDF["Predict"] = y_pred
  print(PredictDF.head().to_string())
  
  return BayesModel, PredictDF

def PredictByBayesian(BayesModel, FeatureSample, TrueLabel, TargetNames):
  if isinstance(FeatureSample, pd.Series):
    FeatureSample = [FeatureSample]
  
  print(FeatureSample)
  
  ColumnNames = ["Proba_" + s for s in TargetNames]
  DirectPredictDF = pd.DataFrame(BayesModel.predict_proba(FeatureSample), columns = ColumnNames)
  DirectPredictDF["LabelTest"] = list(TrueLabel)
  DirectPredictDF["Predict"] = BayesModel.predict(FeatureSample)
  print(DirectPredictDF.head().to_string())
  
  return DirectPredictDF


# In[87]:


FeatureList =["Account length", "Total day minutes", "Total day calls", "Customer service calls"]
TargetAtt = "Churn"
TargetNames = ["False", "True"]
BayesModel, PredictDF = BayesianMLApproach(data_train, label_train, data_test, label_test, FeatureList, TargetAtt,TargetNames)


# In[88]:


FeatureSample = data.loc[[0,3,5,6], FeatureList]
TargetNames = ["False", "True"]
TrueLabel = data.loc[[0,3,5,6], "Churn"]
DirectPredictDF = PredictByBayesian(BayesModel, FeatureSample ,TrueLabel , TargetNames)


# In[ ]:




