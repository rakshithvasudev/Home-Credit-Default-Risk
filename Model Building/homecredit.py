
# coding: utf-8

# # Home Credit Default Risk

# ## Predicting how capable each applicant is of repaying a loan?

# ![home%20credit.jpg](attachment:home%20credit.jpg)

# Introduction: Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.
# 
# Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
# 
# While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.
# 
# https://www.kaggle.com/c/home-credit-default-risk
# 
# The objective of this competition is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. This is a standard supervised classification task:
# 
# Supervised: The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features.
# 
# Classification: The label is a binary variable, 0 (will repay loan on time), 1 (will have difficulty repaying loan)

# ## Import necessary libraries.

# In[2]:


# import numpy for math calculations
import numpy as np

# import pandas for data (csv) manipulation
import pandas as pd

# import matplotlib for plotting
import matplotlib.pyplot as plt

# import seaborn for more plotting options(built on top of matplotlib)
import seaborn as sns


# Supress unnecessary warnings so that the presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# display plots on the notebook itself
get_ipython().magic('matplotlib inline')


# ## Read the data files

# In[3]:


train = pd.read_csv("../dataset/all/application_train.csv")
test = pd.read_csv("../dataset/all/application_test.csv")

new_test = pd.read_csv("../dataset/all/new_test.csv")


# In[4]:


train.info()


# ## How is the statistic?

# In[5]:


train.describe()


# ## How are the target labels spread?

# In[5]:


sns.countplot(train.TARGET)


# In[6]:


train['TARGET'].value_counts()


# ### This is clearly an imbalanced target. There are more number of people who returned - 0 as opposed to people who had difficulties -1. About 91.92 % of applicants repayed!

# ## What are the dimensions of Train and Test dataset?

# In[7]:


print("The train dataset dimensions are as follows: {}".format(train.shape))
print("The test dataset dimensions are as follows: {}".format(test.shape))
print("The test dataset dimensions are as follows: {}".format(new_test.shape))


# ## Look at the train dataset

# In[8]:


train.head()


# ## Look at the test dataset

# In[9]:


test.head()


# ## Look at the New Test dataset

# In[10]:


new_test.head()


# #### As expected, test dataset contains all the columns except the target label.

# ## What are the missing values and their column names?

# In[11]:


def missing_columns(dataframe):
    """
    Returns a dataframe that contains missing column names and 
    percent of missing values in relation to the whole dataframe.
    
    dataframe: dataframe that gives the column names and their % of missing values
    """
    
    # find the missing values
    missing_values = dataframe.isnull().sum().sort_values(ascending=False)
    
    # percentage of missing values in relation to the overall size
    missing_values_pct = 100 * missing_values/len(dataframe)
    
    # create a new dataframe which is a concatinated version
    concat_values = pd.concat([missing_values, missing_values/len(dataframe),missing_values_pct.round(1)],axis=1)

    # give new col names
    concat_values.columns = ['Missing Count','Missing Count Ratio','Missing Count %']
    
    # return the required values
    return concat_values[concat_values.iloc[:,1]!=0]
    


# In[12]:


missing_columns(train)


# In[13]:


missing_columns(test)


# In[14]:


missing_columns(new_test)


# We will have to handle these missing values (known as imputation). Other option would be to drop all those columns where there are large number of missing values. Unless we know the feature importance, it is not possible to make a call on which columns to keep which ones to drop.

# ## What are the different datatypes of columns? - How many floats, integers, categoricals?

# In[15]:


print("Train dataset: \n{}".format(train.dtypes.value_counts()))
print()
print("Test dataset: \n{}".format(test.dtypes.value_counts())) 
print()
print("Test dataset: \n{}".format(new_test.dtypes.value_counts())) 


# #### Turn every column data type of testing set similar to training set. Match datatypes of test in alignment with train. 

# In[6]:


def match_dtypes(training_df,testing_df,target_name='TARGET'):
    """
    This function converts dataframe to match columns in accordance with the 
    training dataframe.
    """
    for column_name in training_df.drop([target_name],axis=1).columns:
         testing_df[column_name]= testing_df[column_name].astype(train[column_name].dtype)
        
    return testing_df
    


# In[7]:


new_test = match_dtypes(train,new_test)


# In[18]:


print("Train dataset: \n{}".format(train.dtypes.value_counts()))
print()
print("Test dataset: \n{}".format(test.dtypes.value_counts())) 
print()
print("Test dataset: \n{}".format(new_test.dtypes.value_counts())) 


# ### In test dataset, 40 int64 indicates that the target label is missing - which is obvious.

# ### What are the different kinds of classes in every categorical column?

# In[19]:


# Number of unique classes in each object column
train.select_dtypes('object').apply(pd.Series.nunique)


# In[20]:


test.select_dtypes('object').apply(pd.Series.nunique)


# In[21]:


new_test.select_dtypes('object').apply(pd.Series.nunique)


# ## Handling Categorical variables - Label Encoding and One Hot Encoding.

# Some machine learning models can't learn if provided with text categories. The categorical variables are to be converted into
# numerical equivalent, which is done by Label encoding and One hot encoding.
# 
# <b>Label encoding:</b> It is the process of assigning each unique category in a categorical variable with an integer. No new columns are created. 

# ![label_encoding.png](attachment:label_encoding.png)

# In[8]:


# Create a label encode object having less than or equal to 2 unique values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
transform_counter = 0

# iterate through all the categorical columns
for col in train.select_dtypes('object').columns:
    
    # select only those columns where number of unique values in the category is less than or equal to 2 
    if pd.Series.nunique(train[col]) <= 2:
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.fit_transform(test[col].astype(str))
        new_test[col] = le.fit_transform(new_test[col].astype(str))

        transform_counter+=1
        
print("Label encoded {} columns.".format(transform_counter))    


# <b>One-hot encoding:</b> create a new column for each unique category in a categorical variable. Each observation recieves a 1 in the column for its corresponding category and a 0 in all other new columns.

# ![one%20hot1.jpg](attachment:one%20hot1.jpg)
# Credit : https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f

# In[9]:


# one-hot encode of categorical variables
train = pd.get_dummies(train,drop_first=True)
test = pd.get_dummies(test,drop_first=True)
new_test = pd.get_dummies(new_test,drop_first=True)


# One hot encoding would added more columns, checking how many there are: 

# In[10]:


print('Training Features shape: ', train.shape)
print('Testing Features shape: ', test.shape)
print('New Testing Features shape: ',new_test.shape)


# There is a mismatch in the count of columns for test and train. This can be fixed by aligning them.

# In[11]:


# collect the target labels to support the aligning 

target = train['TARGET']


# ## Ensure train and test have the same number of columns by aligning.

# In[12]:


train, test = train.align(test,axis=1,join='inner')


# Add the stored target column back into the train dataset.

# In[13]:


train['TARGET'] = target


# Since there are extra columns in the training set and those columns are missing in the new_testing set, let us add those columns and assign them to dummy value of 0.

# In[14]:


def match_columns(training_set,testing_set,target_label='TARGET'):
    """Matches the count of columns from training set to testing set by adding extra cols and setting them to 0."""
    
    for column in training_set.drop([target_label],axis=1).columns:
        if column not in testing_set.columns:
            testing_set[column]=0
    
    return testing_set        


# In[15]:


new_test=match_columns(train,new_test)
new_test.shape


# In[16]:


print('Training Features shape: ', train.shape)
print('Testing Features shape: ', test.shape)
print('Testing Features shape: ', new_test.shape)


# <h3>On the look for Anomalies</h3> 
# </br>
# 
# One problem we always want to be on the lookout for is anomalies within the data. These may be due to mis-typed numbers, errors in measuring equipment, or they could be valid but extreme measurements. One way to support anomalies checking is by looking at the statistics of a column using the describe method. The numbers in the DAYS_BIRTH column are negative because they are recorded relative to the current loan application. To see these stats in years, we can multiply by -1 and divide by the number of days in a year:

# ## How old are clients?

# In[31]:


(train['DAYS_BIRTH']/-365).describe()


# Ages seem to be fine, nothing in particluar seems to be off.

# In[32]:


fig, ax = plt.subplots(figsize =(12,7))
sns.distplot(train['DAYS_BIRTH']/-365,bins=5,kde=False)
plt.xlabel("Age of the client (Years)")


# People in the age range 30-40 years are the most applicants. Which seems pretty normal.

# ### How many years has it been since the applicant started working? 
# The DAYS_EMPLOYED column is negative because the days are relative only to the time of the application. -ve means so many days since the application, the client has been working. +ve means, the client is about to work in those many days. In an ideal world, the -ve has significance, +ve could mean anything from client starts working to client can be fired and resumes working, which in anyway doesn't make sense because the loan might not be given to those clients without any work.

# In[33]:


(train['DAYS_EMPLOYED']/365).describe()


# This doesn't seem right, the maximum value (besides being positive) is about 1000 years!

# ### Who are these special people who got employed 1000 years after issuance of the loan? 

# In[34]:


fig, ax = plt.subplots(figsize=(12,7))
sns.distplot(train['DAYS_EMPLOYED']/365,kde=False)
plt.xlabel("Time before the loan application the persons started current employment(in years)")


# So, how many of these 1000 year anomalies?

# In[17]:


# find the number of records where DAYS_EMPLOYED is between [900,1100] years. 
thousand_anomalies = train[(train['DAYS_EMPLOYED']/365>=900) & (train['DAYS_EMPLOYED']/365<=1100)]
len(thousand_anomalies)


# ## Lets look their ability to repay.

# In[36]:


fig, ax = plt.subplots(figsize=(12,7))
sns.countplot(x='TARGET',data=thousand_anomalies)


# ## Most anomalies were able to repay on time. But how can they be contrasted in relation to non anomalies?

# In[18]:


# get the index of anomalies and non anomalies
anomalies_index = pd.Index(thousand_anomalies.index)
non_anomalies_index = train.index.difference(anomalies_index)


# In[19]:


# get the anomalies records
non_anomalies = train.iloc[non_anomalies_index]


# In[20]:


# get the anomaly targets
anomalies_target = thousand_anomalies['TARGET'].value_counts()
non_anomalies_target = non_anomalies['TARGET'].value_counts()


# In[21]:


# find the default rate for anomalies and non anomalies

print("Anomalies have a default rate of {}%".format(100*anomalies_target[1]/(anomalies_target[1]+anomalies_target[0])))
print("Non Anomalies have a default rate of {}%".format(100*non_anomalies_target[1]/(non_anomalies_target[1]+non_anomalies_target[0])))


# So surprisingly anomalies have lesser default rate!

# Handling the anomalies depends on the exact situation, with no set rules. One of the safest approaches is just to set the anomalies to a missing value and then have them filled in (using Imputation) before machine learning. In this case, since all the anomalies have the exact same value, we want to fill them in with the same value in case all of these loans share something in common. The anomalous values seem to have some importance, so we want to tell the machine learning model if we did in fact fill in these values. As a solution, we will fill in the anomalous values with not a number (np.nan) and then create a new boolean column indicating whether or not the value was anomalous.

# In[22]:


# Create an anomalous flag column
train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
train['DAYS_EMPLOYED'] = train['DAYS_EMPLOYED'].replace({365243: np.nan})


# In[23]:


# Looking at the years employed for anomalies

plt.figure(figsize=(12,8))
(train['DAYS_EMPLOYED']/-365).plot.hist(title = 'Years Employment Histogram')
plt.xlabel("Years worked before application")


# Now it all seems normal!

# In[24]:


# Create an anomalous flag column
test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
test['DAYS_EMPLOYED'] = test['DAYS_EMPLOYED'].replace({365243: np.nan})

# Create an anomalous flag column
new_test['DAYS_EMPLOYED_ANOM'] = new_test["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
new_test['DAYS_EMPLOYED'] = new_test['DAYS_EMPLOYED'].replace({365243: np.nan})


# ## Finding out the most correlated features for the TARGET variable. 

# ## Understanding Correlation
# 
# Correlation is a statistical measure that indicates the extent to which two or more variables fluctuate together. A positive correlation indicates the extent to which those variables increase or decrease in parallel; a negative correlation indicates the extent to which one variable increases as the other decreases.
# 
# A correlation coefficient is a statistical measure of the degree to which changes to the value of one variable predict change to the value of another. When the fluctuation of one variable reliably predicts a similar fluctuation in another variable, there’s often a tendency to think that means that the change in one causes the change in the other. However, correlation does not imply causation. There may be, for example, an unknown factor that influences both variables similarly.
# 
# ![correlation.png](attachment:correlation.png)
# 
# To describe the strength of the
# correlation using the guide that Evans (1996) suggests for the absolute value of r:
# <br/>
#  .00-.19 “very weak”
#  <br/>
#  .20-.39 “weak”
#  <br/>
#  .40-.59 “moderate”
#  <br/>
#  .60-.79 “strong”
#  <br/>
#  .80-1.0 “very strong”
# 
# 
# 
# http://www.statstutor.ac.uk/resources/uploaded/pearsons.pdf <br/>
# https://whatis.techtarget.com/definition/correlation

# In[25]:


corr_train = train.corr()['TARGET']


# ## Looking at the top 10 most positively and negatively correlated features we get:

# In[26]:


print(corr_train.sort_values().tail(10))
corr_train.sort_values().head(10)


# ### Since EXT_SOURCE_3, EXT_SOURCE_2, EXT_SOURCE_1 and DAYS_BIRTH are highly correlated (Relatively), let us also explore the possibility of having them as interaction variables.
# 

# ## Initially filling up the missing values for the most correlated variables.

# In[27]:


from sklearn.preprocessing import Imputer


# In[28]:


poly_fitting_vars = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1','DAYS_BIRTH']


# In[29]:


imputer = Imputer(missing_values='NaN', strategy='median')


# In[30]:


train[poly_fitting_vars] = imputer.fit_transform(train[poly_fitting_vars])


# In[31]:


train[poly_fitting_vars].shape


# In[32]:


test[poly_fitting_vars] = imputer.transform(test[poly_fitting_vars])


# In[33]:


test[poly_fitting_vars].shape


# In[34]:


new_test[poly_fitting_vars] = imputer.transform(new_test[poly_fitting_vars])


# In[35]:


new_test[poly_fitting_vars].shape


# ## Let us generate valuable features - interaction variables.

# In[36]:


from sklearn.preprocessing import PolynomialFeatures


# In[37]:


poly_feat = PolynomialFeatures(degree=4)


# #### I also tried for polynomial degree of order 10. Couldn't find much improvement from degree 4 to 10. Fun fact: order of 10 created over 1000 interaction variables! 

# In[39]:


poly_interaction_train = poly_feat.fit_transform(train[poly_fitting_vars])


# In[40]:


poly_interaction_train.shape


# In[41]:


poly_interaction_test = poly_feat.fit_transform(test[poly_fitting_vars])


# In[42]:


poly_interaction_test.shape


# In[43]:


poly_interaction_new_test = poly_feat.fit_transform(new_test[poly_fitting_vars])


# In[44]:


poly_interaction_new_test.shape


# ## Build a dataframe out of interaction variables only!

# In[45]:


poly_interaction_train = pd.DataFrame(poly_interaction_train,columns=poly_feat.get_feature_names(poly_fitting_vars))


# In[46]:


poly_interaction_train.shape


# In[47]:


poly_interaction_test =  pd.DataFrame(poly_interaction_test,columns=poly_feat.get_feature_names(poly_fitting_vars))


# In[48]:


poly_interaction_test.shape


# In[49]:


poly_interaction_new_test =  pd.DataFrame(poly_interaction_new_test,columns=poly_feat.get_feature_names(poly_fitting_vars))


# In[50]:


poly_interaction_new_test.shape


# ## Add the 'TARGET' column which is later used for looking up correlations with the interaction variables.

# In[51]:


poly_interaction_train['TARGET'] = train['TARGET']


# In[52]:


interaction = poly_interaction_train.corr()['TARGET'].sort_values()


# ## Which are the most correlated interaction variables?

# In[53]:


# looking at the top 15 most positive and negative correlated interaction variables.
print(interaction.tail(15))
(interaction.head(15))


# ## Get the names of the columns which have the highest correlation - '1' and 'TARGET' can be dropped.

# In[54]:


set(interaction.head(15).index).union(interaction.tail(15).index).difference(set({'1','TARGET'}))


# ## Choose the selected columns which have highest correlation to 'TARGET'. Columns '1' and 'TARGET' are not necessary!

# In[55]:


selected_inter_variables = list(set(interaction.head(15).index).union(interaction.tail(15).index).difference(set({'1','TARGET'})))


# In[56]:


# look at the selected features
poly_interaction_train[selected_inter_variables].head()


# In[57]:


poly_interaction_test[selected_inter_variables].head()


# In[58]:


poly_interaction_new_test[selected_inter_variables].head()


# ## Get a list of unselected columns that are to be dropped.

# In[59]:


unselected_cols = [element for element in poly_interaction_train.columns if element not in selected_inter_variables]


# ##  Drop the unselected columns of the interaction dataframes - train and test versions both.

# In[60]:


poly_interaction_train = poly_interaction_train.drop(unselected_cols,axis=1)


# In[61]:


poly_interaction_test = poly_interaction_test.drop(list(set(unselected_cols).difference({'TARGET'})),axis=1)


# In[62]:


poly_interaction_new_test = poly_interaction_new_test.drop(list(set(unselected_cols).difference({'TARGET'})),axis=1)


# ## Merge polynomial features into the original dataframes using their indices.

# #### Dropping columns 'EXT_SOURCE_2' and 'EXT_SOURCE_3' since they're already present in the source dataset.

# In[63]:


train = train.join(poly_interaction_train.drop(['EXT_SOURCE_2', 'EXT_SOURCE_3'],axis=1))


# In[64]:


test = test.join(poly_interaction_test.drop(['EXT_SOURCE_2', 'EXT_SOURCE_3'],axis=1))


# In[65]:


new_test = new_test.join(poly_interaction_new_test.drop(['EXT_SOURCE_2', 'EXT_SOURCE_3'],axis=1))


# ## What are their merged dataframe dimensions?

# In[66]:


print("The train dataset dimensions are as follows: {}".format(train.shape))
print("The test dataset dimensions are as follows: {}".format(test.shape))
print("The new test dataset dimensions are as follows: {}".format(new_test.shape))


# # Domain Feature Engineering

# ### Industry expert opinion and metrics.
# 
# ![industry%20expert.jpg](attachment:industry%20expert.jpg)
# 
# This article from Wells Fargo explains what factors are looked at while providing money to borrowers.
# https://www.wellsfargo.com/financial-education/credit-management/five-c/
# 
# Here are the major factors accordingly:<br>
# <b>Credit history:</b> Qualifying for the different types of credit hinges largely on your credit history — the track record you’ve established while managing credit and making payments over time. Your credit report is primarily a detailed list of your credit history, consisting of information provided by lenders that have extended credit to you. While information may vary from one credit reporting agency to another, the credit reports include the same types of information, such as the names of lenders that have extended credit to you, types of credit you have, your payment history, and more. 
# 
# In addition to the credit report, lenders may also use a credit score that is a numeric value – usually between 300 and 850 – based on the information contained in your credit report. The credit score serves as a risk indicator for the lender based on your credit history. Generally, the higher the score, the lower the risk. Credit bureau scores are often called "FICO® scores" because many credit bureau scores used in the U.S. are produced from software developed by Fair Isaac Corporation (FICO). While many lenders use credit scores to help them make their lending decisions, each lender has its own criteria, depending on the level of risk it finds acceptable for a given credit product.
# 
# <b>Capacity:</b> Lenders need to determine whether you can comfortably afford your payments. Your income and employment history are good indicators of your ability to repay outstanding debt. Income amount, stability, and type of income may all be considered. The ratio of your current and any new debt as compared to your before-tax income, known as debt-to-income ratio (DTI), may be evaluated.
# 
# <b>Collateral (when applying for secured loans):</b> Loans, lines of credit, or credit cards you apply for may be secured or unsecured. With a secured product, such as an auto or home equity loan, you pledge something you own as collateral. The value of your collateral will be evaluated, and any existing debt secured by that collateral will be subtracted from the value. The remaining equity will play a factor in the lending decision.
# 
# <b>Capital:</b> While your household income is expected to be the primary source of repayment, capital represents the savings, investments, and other assets that can help repay the loan. This can be helpful if you lose your job or experience other setbacks.
# 
# <b>Conditions:</b> Lenders may want to know how you plan to use the money and will consider the loan’s purpose, such as whether the loan will be used to purchase a vehicle or other property. Other factors, such as environmental and economic conditions, may also be considered. 
# 
# Since we don't consider credit history, we can asses other 4 C's.
# 
# Let us incorporate the following variables: 
# 
# 1) <b>debt-to-income ratio(DIR) = Credit amount of the loan / Total Income = AMT_CREDIT/AMT_INCOME_TOTAL</b><br/>
# 2) <b>annuity-to-income ratio(AIR) = Loan annuity / Total Income = AMT_ANNUITY/AMT_INCOME_TOTAL</b><br/>
# 3) <b>annuity-to-credit ratio(ACR) = Loan annuity/ Credit amount of the loan = AMT_ANNUITY/AMT_CREDIT</b><br/>
# 4) <b>days-employed-to-age ratio(DAR) = Number of days employed/ Age of applicant = DAYS_EMPLOYED/DAYS_BIRTH</b><br/>
# 

# In[67]:


train['DIR'] = train['AMT_CREDIT']/train['AMT_INCOME_TOTAL']
train['AIR'] = train['AMT_ANNUITY']/train['AMT_INCOME_TOTAL']
train['ACR'] = train['AMT_ANNUITY']/train['AMT_CREDIT']
train['DAR'] = train['DAYS_EMPLOYED']/train['DAYS_BIRTH']


# In[68]:


test['DIR'] = test['AMT_CREDIT']/test['AMT_INCOME_TOTAL']
test['AIR'] = test['AMT_ANNUITY']/test['AMT_INCOME_TOTAL']
test['ACR'] = test['AMT_ANNUITY']/test['AMT_CREDIT']
test['DAR'] = test['DAYS_EMPLOYED']/test['DAYS_BIRTH']


# In[69]:


new_test['DIR'] = new_test['AMT_CREDIT']/new_test['AMT_INCOME_TOTAL']
new_test['AIR'] = new_test['AMT_ANNUITY']/new_test['AMT_INCOME_TOTAL']
new_test['ACR'] = new_test['AMT_ANNUITY']/new_test['AMT_CREDIT']
new_test['DAR'] = new_test['DAYS_EMPLOYED']/new_test['DAYS_BIRTH']


# ## Look at the correlation of the newly added variables in relation to the 'TARGET'

# In[70]:


corr_vals = train.corr()['TARGET']


# In[71]:


corr_vals.tail(4)


# ## Hmmm, not much correlation - Linear!

# # Preparing the dataset for feeding into the model.

# ## Feature Imputing

# Feature imputation is the process of filling up missed/NAN values for those columns where 
# certain cells are not filled by default due to reasons such as outlier replacement / unavailable data 
# or incorrect entires during capturing the data.   

# In[72]:


from sklearn.preprocessing import MinMaxScaler, Imputer


# In[73]:


features = list(set(train.columns).difference({'TARGET'}))


# Imputation is done for the median value of every column.

# In[74]:


imputer = Imputer(strategy="median")


# ## Feature Scaling
# 
# Feature scaling is a method used to standardize the range of independent variables or features of data. 
# In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.
#  
# 
# Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. 
# For example, the majority of classifiers calculate the distance between two points by the Euclidean distance. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.
# Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.

# In[75]:


new_test = new_test.replace(to_replace=np.inf,value=0)


# In[76]:


scaler = MinMaxScaler(feature_range = (0, 1))


# In[77]:


imputer.fit(train.drop(['TARGET'],axis=1))


# In[78]:


train_transformed = imputer.transform(train.drop(['TARGET'],axis=1))


# In[79]:


test_transformed = imputer.transform(test)


# In[80]:


new_test_transformed = imputer.transform(new_test)


# In[81]:


train_transformed = scaler.fit_transform(train_transformed)


# In[82]:


test_transformed = scaler.transform(test_transformed)


# In[83]:


new_test_transformed = scaler.transform(new_test_transformed)


# In[84]:


# new_test[new_test.isnull().any(axis=1)]


# In[85]:


print("The train dataset dimensions are as follows: {}".format(train_transformed.shape))
print("The test dataset dimensions are as follows: {}".format(test_transformed.shape))
print("The new test dataset dimensions are as follows: {}".format(new_test_transformed.shape))


# In[89]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pcc = pca.fit_transform(new_test_transformed)
pcc = pd.DataFrame(pcc , columns=['PC1','PC2'])


# # Split the dataset into training set and validation set

# In[104]:


from sklearn.model_selection import train_test_split

X_training_set, X_validation_set, y_training_set, y_validation_set = train_test_split(train_transformed, 
                                                                                      target, test_size=0.33, random_state=42)


# ## Use the Model

# ### 1. Logistic Regression

# Logistic Regression measures the relationship between the dependent variable (target label to predict) and the one or more independent variables (features), by estimating probabilities using it’s underlying logistic function.
# 
# These probabilities must then be transformed into binary values in order to actually make a prediction. This is the task of the logistic function, also called the sigmoid function. The Sigmoid-Function is an S-shaped curve that can take any real-valued number and map it into a value between the range of 0 and 1, but never exactly at those limits. This values between 0 and 1 will 
# then be transformed into either 0 or 1 using a threshold classifier.
# 
# ![logistic%20function.png](attachment:logistic%20function.png)
# 
# 
# We want to maximize the likelihood that a random data point gets classified correctly, which is called Maximum Likelihood Estimation. Maximum Likelihood Estimation is a general approach to estimating parameters in statistical models. You can maximize the likelihood using different methods like an optimization algorithm such as gradient descent or Newton's method.
# 
# ![linearly_separable_log_reg.png](attachment:linearly_separable_log_reg.png)
# 
# Image source : http://blog.sairahul.com/2014/01/linear-separability.html <br/>
# More explanation here : https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf <br/>
# Intuitive breakdown here : https://machinelearningmastery.com/logistic-regression-for-machine-learning/

# In[105]:


# Starting with Logistic Regression.

from sklearn.linear_model import LogisticRegression

logistic_regressor = LogisticRegression(C = 2)


# In[106]:


logistic_regressor.fit(X_training_set,y_training_set)


# In[107]:


log_regression_pred = logistic_regressor.predict(X_validation_set)


# In[165]:


logistic_new = logistic_regressor.predict(new_test_transformed)


# In[180]:


pd.DataFrame({'target':logistic_new})['target'].value_counts()


# # Understanding Accuracy metrics
# 
# <b>1.True Positives (TP):</b> True positives are the cases when the actual class of the data point was 1(True) and the predicted is also 1(True)
# Ex: The case where a person is actually having cancer(1) and the model classifying his case as cancer(1) comes under True positive.
# 
# <b>2.True Negatives (TN):</b> True negatives are the cases when the actual class of the data point was 0(False) and the predicted is also 0(False
# 
# Ex: The case where a person NOT having cancer and the model classifying his case as Not cancer comes under True Negatives.
# 
# <b>3.False Positives (FP):</b> False positives are the cases when the actual class of the data point was 0(False) and the predicted is 1(True). False is because the model has predicted incorrectly and positive because the class predicted was a positive one. (1)
# 
# Ex: A person NOT having cancer and the model classifying his case as having cancer comes under False Positives.
# 
# <b>4.False Negatives (FN):</b> False negatives are the cases when the actual class of the data point was 1(True) and the predicted is 0(False). False is because the model has predicted incorrectly and negative because the class predicted was a negative one. (0)
# 
# Ex: A person having cancer and the model classifying his case as No-cancer comes under False Negatives.
# 
# ### Minimization and Trade offs :
# 
# We know that there will be some error associated with every model that we use for predicting the true class of the target variable. This will result in False Positives and False Negatives(i.e Model classifying things incorrectly as compared to the actual class).
# 
# There’s no hard and fast rule that says what should be minimised in all the situations. It purely depends on the business needs and the context of the problem you are trying to solve. Based on that, we might want to minimise either False Positives or False negatives.
# 
# ### Accuracy:
# Accuracy in classification problems is the number of correct predictions made by the model over all kinds predictions made.
# ![accuracy.png](attachment:accuracy.png)
# 
# 
# ### Precision:
# Precision talks about how precise/accurate the model is out of those predicted positive, how many of them are actual positive.
# ![precision.png](attachment:precision.png)
# 
# 
# ### Recall - True Positive Rate:
# What percent of the positive cases did the model catch (predicted positive) amongst all positive cases. Recall actually calculates how many of the Actual Positives our model capture through labeling it as Positive.
# ![recall.png](attachment:recall.png)
# 
# 
# ### False Positive Rate:
# <b>False Positive Rate = False Positives / (False Positives + True Negatives) </b>
# 
# 
# ### F-1 Score:
# F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
# ![f-1%20score.png](attachment:f-1%20score.png)
# 
# 
# ### ROC (receiver operating characteristic) Curve:
# A curve of true positive rate vs. false positive rate at different classification thresholds.
# 
# ### AUROC (Area under ROC):
# An evaluation metric that considers all possible classification thresholds.
# 
# The Area Under the ROC curve is the probability that a classifier will be more confident that a randomly chosen positive example is actually positive than that a randomly chosen negative example is positive.
# 
# image source : https://medium.com/greyatom/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b
# 
# https://en.wikipedia.org/wiki/F1_score
# 
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

# In[109]:


from sklearn.metrics import accuracy_score,classification_report, roc_auc_score
print("The accuracy in general is : ", accuracy_score(y_validation_set,log_regression_pred))
print("\n")
print("The classification report is as follows:\n", classification_report(y_validation_set,log_regression_pred))
print("ROC AUC score is: ",roc_auc_score(y_validation_set,log_regression_pred))


# We want to predict the probabilty of not paying a loan, so we use the model predict.proba method. 
# This returns an m x 2 array where m is the number of datapoints.
# The first column is the probability of the target being 0 and the second column is the probability of the 
# target being 1. We want the probability the loan is not repaid, so we will select the second column.

# In[110]:


log_regression_pred_test = logistic_regressor.predict_proba(test_transformed)


# In[111]:


# selecting the second column
log_regression_pred_test[:,1]


# In[112]:


submission_log_regression = test[['SK_ID_CURR']]
submission_log_regression['TARGET'] = log_regression_pred_test[:,1]


# In[113]:


submission_log_regression.head(10)


# In[114]:


submission_log_regression.to_csv("log_regression.csv",index=False)


# Scored  0.732 in AUROC - Eval by Kaggle.

# ### 2. Random Forest - Bagging ensemble of Decision trees

# A decision tree is a Machine Learning algorithm capable of fitting complex datasets and performing both classification and regression tasks. The idea behind a tree is to search for a pair of variable-value within the training set and split it in such a way that will generate the "best" two child subsets. The goal is to create branches and leafs based on an optimal splitting criteria, a process called tree growing. Specifically, at every branch or node, a conditional statement classifies the data point based on a fixed threshold in a specific variable, therefore splitting the data. To make predictions, every new instance starts in the root node (top of the tree) and moves along the branches until it reaches a leaf node where no further branching is possible.
# 
# ![random%20forest.png](attachment:random%20forest.png)
# image source: https://www.kdnuggets.com/2017/10/random-forests-explained.html
# 
# Random Forests are trained via the bagging method. Bagging or Bootstrap Aggregating, consists of randomly sampling subsets of the training data, fitting a model to these smaller data sets, and aggregating the predictions. This method allows several instances to be used repeatedly for the training stage given that we are sampling with replacement. Tree bagging consists of sampling subsets of the training set, fitting a Decision Tree to each, and aggregating their result.
# 
# In relation to sklearn: 
# 
# 
# A random forest is a collection of random decision trees (of number n_estimators in sklearn). What you need to understand is how to build one random decision tree.
# 
# Roughly speaking, to build a random decision tree you start from a subset of your training samples. At each node you will draw randomly a subset of features (number determined by max_features in sklearn). For each of these features you will test different thresholds and see how they split your samples according to a given criterion (generally entropy or gini, criterion parameter in sklearn). Then you will keep the feature and its threshold that best split your data and record it in the node. When the construction of the tree ends (it can be for different reasons: maximum depth is reached (max_depth in sklearn), minimum sample number is reached (min_samples_leaf in sklearn) etc.) you look at the samples in each leaf and keep the frequency of the labels. As a result, it is like the tree gives you a partition of your training samples according to meaningful features.
# 
# As each node is built from features taken randomly, you understand that each tree built in this way will be different. This contributes to the good compromise between bias and variance.
# 
# Then in testing mode, a test sample will go through each tree, giving you label frequencies for each tree. The most represented label is generally the final classification result.
# 
# https://stackoverflow.com/questions/31344732/a-simple-explanation-of-random-forest<br>
# Original paper: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf

# In[115]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators = 500, random_state = 50, verbose = 1, n_jobs = -1)


# In[116]:


random_forest.fit(X_training_set,y_training_set)


# In[117]:


random_forest_pred = random_forest.predict(X_validation_set)


# In[118]:


from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
print("The accuracy in general is : ", accuracy_score(y_validation_set,random_forest_pred))
print("\n")
print("The classification report is as follows:\n", classification_report(y_validation_set,random_forest_pred))
print("ROC AUC score is: ",roc_auc_score(y_validation_set,random_forest_pred))


# In[119]:


random_forest_pred_test = random_forest.predict_proba(test_transformed)


# In[166]:


random_forest_new = random_forest.predict(new_test_transformed)


# In[182]:


pd.DataFrame({'target':random_forest_new})['target'].value_counts()


# In[121]:


submission_random_forest = test[['SK_ID_CURR']]
submission_random_forest['TARGET'] = random_forest_pred_test[:,1]


# In[122]:


submission_random_forest.to_csv("random_forest.csv",index=False)


# Scored  0.679 in AUROC - Eval by Kaggle.

# ### Feature importance of random forest classifier

# In[123]:


# build a dataframe for checking out feature importance


# In[124]:


feature_importance_df = pd.DataFrame({'Feature':features,'Importance':random_forest.feature_importances_})


# In[125]:


def plot_importance(df):
    """
    Builds the dataset to plot the feature importance.
    
    """
    # Sort features according to importance
    df = df.sort_values(['Importance'],ascending=False).reset_index()
    
    # drop the old index to avoid confusion
    df = df.drop(['index'],axis=1)
    
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 9))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:30]))), 
            df['Importance'].head(30), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:30]))))
    ax.set_yticklabels(df['Feature'].head(30))
    
    plt.xlabel("Normalized feature importance")
    plt.ylabel("Features")
    
    plt.show()
    return df


# ##  Some of the interaction variables made it to the top 10 of the feature importance plot.

# In[126]:


sorted_importance = plot_importance(feature_importance_df)


# # What are the top 20 features?

# In[127]:


sorted_importance.head(20)


# # What happened to the domain engineered features?

# In[128]:


sorted_importance[(sorted_importance.Feature=='DIR')|
                  (sorted_importance.Feature=='AIR')|
                  (sorted_importance.Feature=='ACR')|
                  (sorted_importance.Feature=='DAR')]


# Unfortunately, they didn't end up on the top 10!

# ## 3. Extreme Gradient Boost Model

# XGBoost Tree boosting is a highly effective and widely used machine learning method.
# The library is laser focused on computational speed and model performance, as such there are few frills. Nevertheless, it does offer a number of advanced features.
# 
# ### Model Features
# The implementation of the model supports the features of the scikit-learn and R implementations, with new additions like regularization. Three main forms of gradient boosting are supported:
# 
# <b>Gradient Boosting algorithm also called gradient boosting machine including the learning rate. </b><br>
# <b>Stochastic Gradient Boosting with sub-sampling at the row, column and column per split levels. </b><br>
# <b>Regularized Gradient Boosting with both L1 and L2 regularization.</b></br>
# 
# ### System Features
# 
# The library provides a system for use in a range of computing environments, not least:
# Parallelization of tree construction using all of your CPU cores during training.<br>
# Distributed Computing for training very large models using a cluster of machines.<br>
# Out-of-Core Computing for very large datasets that don’t fit into memory.<br>
# Cache Optimization of data structures and algorithm to make best use of hardware.<br>
# 
# ### Algorithm Features
# 
# The implementation of the algorithm was engineered for efficiency of compute time and memory resources. A design goal was to make the best use of available resources to train the model. Some key algorithm implementation features include:
# 
# <b>Sparse Aware implementation with automatic handling of missing data values.</b><br>
# <b>Block Structure to support the parallelization of tree construction.</b><br>
# <b>Continued Training so that you can further boost an already fitted model on new data</b><br>
# 
# 
# ### Benchmark
# 
# ![Benchmark-Performance-of-XGBoost.png](attachment:Benchmark-Performance-of-XGBoost.png)
# Image source: http://datascience.la/benchmarking-random-forest-implementations/
# 
# 
# ### Gradient Boosting
# Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.
# 
# 
# ### Level-wise Tree Growth
# Tree growth is levelwise as shown: 
# 
# ![level%20wise.png](attachment:level%20wise.png)
# image source: https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
# 
# 
# Link to Paper: http://delivery.acm.org/10.1145/2940000/2939785/p785-chen.pdf?ip=24.180.58.36&id=2939785&acc=CHORUS&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1534839698_b4c1981ee9aff25a9ef1d8394ff10dea
# 
# Original PPT from founder: https://speakerdeck.com/datasciencela/tianqi-chen-xgboost-overview-and-latest-news-la-meetup-talk

# In[129]:


from xgboost import XGBClassifier


# In[130]:


xgb_classifier = XGBClassifier(n_estimators=250,max_depth=5)


# In[131]:


xgb_classifier.fit(X_training_set,y_training_set)


# In[132]:


xgb_pred = xgb_classifier.predict(X_validation_set)


# In[167]:


xgb_new = xgb_classifier.predict(new_test_transformed)


# In[183]:


pd.DataFrame({'target':xgb_new})['target'].value_counts()


# In[134]:


from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
print("The accuracy in general is : ", accuracy_score(y_validation_set,xgb_pred))
print("\n")
print("The classification report is as follows:\n", classification_report(y_validation_set,xgb_pred))
print("ROC AUC score is: ",roc_auc_score(y_validation_set,xgb_pred))


# In[135]:


xgb_pred_test = xgb_classifier.predict_proba(test_transformed)


# In[136]:


submission_xgb = test[['SK_ID_CURR']]
submission_xgb['TARGET'] = xgb_pred_test[:,1]


# In[137]:


submission_xgb.to_csv("xgb.csv",index=False)


# Scored  0.736 in AUROC - Eval by Kaggle.

# ## XGBoost Feature Importance

# In[138]:


xgb_feature_importance_df = pd.DataFrame({'Feature':features,'Importance':xgb_classifier.feature_importances_})


# In[139]:


sorted_importance = plot_importance(xgb_feature_importance_df)


# ## What is the rank of Domain engineered features on this model?

# In[140]:


sorted_importance[(sorted_importance.Feature=='DIR')|
                  (sorted_importance.Feature=='AIR')|
                  (sorted_importance.Feature=='ACR')|
                  (sorted_importance.Feature=='DAR')]


# # What are the top 20 features?

# In[141]:


sorted_importance.head(20)


# ## 4. Microsoft's LightGBM

# Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks.
# 
# Since it is based on decision tree algorithms, it splits the tree leaf wise with the best fit whereas other boosting algorithms split the tree depth wise or level wise rather than leaf-wise. So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. Also, it is surprisingly very fast, hence the word ‘Light’.
# 
# ![leaf%20wise.png](attachment:leaf%20wise.png)
# source: https://lightgbm.readthedocs.io/en/latest/Features.html
# 
# Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm. 
# 
# Leaf wise splits lead to increase in complexity and may lead to overfitting and it can be overcome by specifying another parameter max-depth which specifies the depth to which splitting will occur.
# 

# In[142]:


import lightgbm as lgb


# In[143]:


lgb_classifier = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        learning_rate=0.1, max_depth=-1, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
        n_jobs=-1, num_leaves=40, objective=None, random_state=None,
        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=0)


# In[144]:


lgb_classifier.fit(X_training_set,y_training_set)


# In[145]:


lgb_pred = lgb_classifier.predict(X_validation_set)


# In[168]:


lgb_new = lgb_classifier.predict(new_test_transformed)


# In[184]:


pd.DataFrame({'target':lgb_new})['target'].value_counts()


# ### What is the accuracy score on LightGBM?

# In[147]:


from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
print("The accuracy in general is : ", accuracy_score(y_validation_set,lgb_pred))
print("\n")
print("The classification report is as follows:\n", classification_report(y_validation_set,lgb_pred))
print("ROC AUC score is: ",roc_auc_score(y_validation_set,lgb_pred))


# In[148]:


lgb_feature_importance_df = pd.DataFrame({'Feature':features,'Importance':lgb_classifier.feature_importances_/np.sum(lgb_classifier.feature_importances_)})


# In[149]:


sorted_importance = plot_importance(lgb_feature_importance_df)


# ## What is the rank of Domain engineered features on this model?

# In[150]:


sorted_importance[(sorted_importance.Feature=='DIR')|
                  (sorted_importance.Feature=='AIR')|
                  (sorted_importance.Feature=='ACR')|
                  (sorted_importance.Feature=='DAR')]


# # What are the top 20 features?

# In[151]:


sorted_importance.head(20)


# ## 5. Naive Bayes Classifier 

# It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.
# 
# ![bayesrule.jpg](attachment:bayesrule.jpg)
# 
# Case example:
# ![Bayes-rule__20.png](attachment:Bayes-rule__20.png)
# https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
# 
# https://www.geeksforgeeks.org/naive-bayes-classifiers/

# In[152]:


from sklearn.naive_bayes import GaussianNB


# In[153]:


bayes_class = GaussianNB()


# In[154]:


bayes_class.fit(X_training_set,y_training_set)


# In[155]:


bayes_preds = bayes_class.predict(X_validation_set)


# In[169]:


bayes_new = bayes_class.predict(new_test_transformed)


# In[185]:


pd.DataFrame({'target':bayes_new})['target'].value_counts()


# In[157]:


from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
print("The accuracy in general is : ", accuracy_score(y_validation_set,bayes_preds))
print("\n")
print("The classification report is as follows:\n", classification_report(y_validation_set,bayes_preds))
print("ROC AUC score is: ",roc_auc_score(y_validation_set,bayes_preds))


# #### Accuracy is low, while AUC is large!

# In[158]:


bayes_rule_pred_test = bayes_class.predict_proba(test_transformed)


# In[159]:


submission_nb = test[['SK_ID_CURR']]
submission_nb['TARGET'] = bayes_rule_pred_test[:,1]
submission_nb.to_csv("nb.csv",index=False)


# ## 6. Incorporating Ensemble Modeling

# ### So What is Ensemble Modeling?
# https://blog.statsbot.co/ensemble-learning-d1dcd548e936

# Ensemble model combines multiple individual models together and delivers superior prediction power. Basically, an ensemble is a supervised learning technique for combining multiple weak learners/ models to produce a strong learner. Ensemble model works better, when we ensemble models with low correlation.
# 
# 

# ## Bagging - Voting
# 
# Bagging stands for bootstrap aggregation. One way to reduce the variance of an estimate is to average together multiple estimates. For example, we can train M different trees on different subsets of the data (chosen randomly with replacement) and compute the ensemble:
# 
# 

# ![bagging.png](attachment:bagging.png)

# Bagging uses bootstrap sampling to obtain the data subsets for training the base learners. For aggregating the outputs of base learners, bagging uses voting for classification and averaging for regression.

# ## Boosting - Weight updates 
# Boosting refers to a family of algorithms that are able to convert weak learners to strong learners. The main principle of boosting is to fit a sequence of weak learners− models that are only slightly better than random guessing, such as small decision trees− to weighted versions of the data. More weight is given to examples that were misclassified by earlier rounds.
# 
# The predictions are then combined through a weighted majority vote (classification) or a weighted sum (regression) to produce the final prediction. The principal difference between boosting and the committee methods, such as bagging, is that base learners are trained in sequence on a weighted version of the data.

# ## Image difference between bagging and boosting
# 
# ![bagging%20vs%20boosting.png](attachment:bagging%20vs%20boosting.png)
# https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/

# ### Stacking
# Stacking is an ensemble learning technique that combines multiple classification or regression models via a meta-classifier or a meta-regressor. The base level models are trained based on a complete training set, then the meta-model is trained on the outputs of the base level model as features.
# 
# The base level often consists of different learning algorithms and therefore stacking ensembles are often heterogeneous. 

# ![stacking.png](attachment:stacking.png)

# ## Building a stacked model - Voting Approach (Bagging)

# In[160]:


from scipy import stats
def stacked_model(X_training_set):
    """
    This method performs the stacked ensambling of all the models - XGBoost, LGBoost, Random forest, 
    Naive Bayes,Logistic Regression.
    """  
    stacked_predictions = np.array([])

    for element in X_training_set:
         stacked_predictions = np.append(stacked_predictions,stats.mode(element)[0][0])

    return stacked_predictions


# ### Combine all the test results into a multidimensional array to feed into the stacked model.

# In[161]:


combined_array = (pd.DataFrame({'LR':log_regression_pred,
                                'XGB':xgb_pred,
                                'LGB':lgb_pred,
                                'RF':random_forest_pred,
                                'Bayes':bayes_preds}).values)


# In[173]:


combined_new = (pd.DataFrame({'LR':logistic_new,
                                'XGB':xgb_new,
                                'LGB':lgb_new,
                                'RF':random_forest_new,
                                'Bayes':bayes_new}).values)


# ### Make Predictions from the stacked model.

# In[162]:


stacked_model_pred = stacked_model(combined_array)


# In[186]:


stacked_new = stacked_model(combined_new).astype(int)


# In[187]:


pd.DataFrame({'target':stacked_new})['target'].value_counts()


# ### How is the Accuracy?

# In[163]:


from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
print("The accuracy in general is : ", accuracy_score(y_validation_set,stacked_model_pred))
print("\n")
print("The classification report is as follows:\n", classification_report(y_validation_set,stacked_model_pred))
print("ROC AUC score is: ",roc_auc_score(y_validation_set,stacked_model_pred))


# In[164]:


new_test.head()

