# Importing required Library
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
import pickle


# =============================================================================
# WORKING WITH TRAINING DATA
# =============================================================================

# Methods
def classifiy_features(data, columns, yr_features):
    """
    Classifies the Train data Features based on its type
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
    columns : TYPE
        List.
    yr_features : TYPE
        List.

    Returns
    -------
    None.

    """
    global numeric_features
    global cate_features
    
    numeric_features.clear()  
    cate_features.clear()
    
    for col in columns :
        d_type = data[col].dtype
        if (((d_type == np.int64) or (d_type == np.float64) or 
             (d_type == np.int32))
        and yr_features.__contains__(col) != True) :
            numeric_features.append(col)
        elif (((d_type == np.object) or (d_type == np.str)  or 
             (d_type == np.float)) and 
              yr_features.__contains__(col) != True):
            cate_features.append(col)
            

def classifiy_test_features(data, columns, yr_features):
    """
    Classifies the Test data Features based on its type
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
    columns : TYPE
        List.
    yr_features : TYPE
        List.

    Returns
    -------
    None.

    """
    global test_numeric_features
    global test_cate_features
    
    test_numeric_features.clear()  
    test_cate_features.clear()
    
    for col in columns :
        d_type = data[col].dtype
        if (((d_type == np.int64) or (d_type == np.float64) or 
             (d_type == np.int32))
        and yr_features.__contains__(col) != True) :
            test_numeric_features.append(col)
        elif (((d_type == np.object) or (d_type == np.str)  or 
             (d_type == np.float)) and 
              yr_features.__contains__(col) != True):
            test_cate_features.append(col)
            
            
            
def get_column_names(columns):
    """
    Returns the column names

    Parameters
    ----------
    columns : TYPE
        List.

    Returns
    -------
    col_list : TYPE
        List.

    """
    col_list = []
    for col in columns:
        col_list.append(col)
    return col_list



def update_year_into_category(data, yr_features):
    """
    Converts Years into Categories
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
        
    yr_features : TYPE
        string.

    Returns
    -------
    data : TYPE
        Dataframe
        
    """
    temp_df = data.copy()
    
    for col in yr_features :
        temp_df.loc[(temp_df[col] >= 1965) & (temp_df[col] <= 2010), col] = 1
        temp_df.loc[(temp_df[col] >= 1919) & (temp_df[col] <= 1964), col] = 2
        temp_df.loc[(temp_df[col] >= 1872) & (temp_df[col] <= 1918), col] = 3
        data[col] = temp_df[col]
    for col in year_features :
        data[col] = data[col].astype(int)
        data[col] = data[col].astype(str)
    return data
        

def get_null_percentage(data, features) :
    """
    Returns the null percentage of the features
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
        
    features : TYPE
        string.
        
    Returns
    -------
    null_percent : TYPE
        int.
    
    """
    null_percent = data[features].isnull().mean() * 100
    null_percent = null_percent.sort_values(ascending=False)
    return null_percent

def replace_null_values(data, features) :
    """
    Replaces the null vales into its MODE value
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
        
    features : TYPE
        List.

    Returns
    -------
    None.
    
    """
    for feature in features :
        null_replace = data[feature].mode()[0]
        print(feature, '------', null_replace)
        data.loc[data[feature].isnull(), feature] = null_replace
        
        
def get_outlier_details(data, col_name):
    """
    Collects the outlier details
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
        
    col_name : TYPE
        string.

    Returns
    -------
    None.
    
    """
    data.sort_values(col_name, ascending = True, inplace=True)
    q1 = data[col_name].quantile(0.25)
    q3 = data[col_name].quantile(0.75)
    iqr = q3-q1
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    print(data[col].describe())
    print('min: ', fence_low, ', max:', fence_high) 
    print('--------------------------------------------------------')
    

def remove_outlier(data, col_name):
    """
    Removes the outlier
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
        
    col_name : TYPE
        string.

    Returns
    -------
    data : TYPE
        Series.
    
    """
    data.sort_values(col_name, ascending = True, inplace=True)
    q1 = data[col_name].quantile(0.25)
    q3 = data[col_name].quantile(0.75)
    iqr = q3 - q1
    fence_low  = q1 - (1.5*iqr)
    fence_high = q3 + (1.5*iqr)
    median = data[col].median()
    print(col,'--', fence_low,'--', fence_high,'--', median)
    if (fence_low != 0 and fence_high != 0):
        print('entered')
        data[col] = data.loc[((data[col_name] > fence_low) & 
                              (data[col_name] < fence_high)), col]
        data[col].fillna(median, inplace=True)
    return data[col]


def desc_features(data, features):
    """
    Describes the featurs
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
        
    features : TYPE
        List.

    Returns
    -------
    None.
    
    """
    for feat in features:
        print(data[feat].describe())
        
        
        
def get_uniques(data, feature):
    """
    Gets unique data in the featurs
    
    Parameters
    ----------
    data : TYPE
        Dataframe.
        
    features : TYPE
        string.

    Returns
    -------
    None.
    
    """
    uniques = data[feature].sort_values().unique()
    print(uniques)
    print(len(uniques))


# Data Preprocessing
        
data_set = pd.read_csv('C:/Users/SRIKANDAN/DataScience/Project/HousingPricePrediction/train.csv', 
                       index_col = 'Id')
data_set.info()

data_set_shape = data_set.shape


corr = data_set.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
sns.heatmap(corr)


"""
    Features having more than 70% correlation
    corr -> 0.7 - 1.0 is a Strong correlated featrues
"""
index = 0
for col in corr.columns:
    print(col, '---',index)
    print(corr.loc[(corr[col] > 0.7 ) & 
                              (corr[col] != 1.0), col])
    index += 1
    
""" 
    Correlated features :  YearBuilt - GarageYrBlt, GarageCars - GarageArea,
    So we are droping above correlated features 
""" 
corr_drop_cols = ['GarageYrBlt', 'GarageCars']
data_set.drop(corr_drop_cols, axis=1, inplace=True)


# Classifying Features
data_set_cols = get_column_names(data_set.columns)

numeric_features = []
cate_features = []
year_features = ['YearBuilt', 'YearRemodAdd']
classifiy_features(data_set, data_set.columns, year_features)


"""
    Updating DateTime Values
    Idead : Converting Year to Categorical Values with 5 Categories
    1   -> 1988 - 1987
    2   -> 1942 - 1941
    3   -> 1900 - 1918
    0   -> For Nan
"""
data_set = update_year_into_category(data_set, ['YearBuilt', 'YearRemodAdd'])
data_set.drop(['YrSold'], axis=1, inplace=True)


#  Updating Catrgorical Values
cat_null_percent = get_null_percentage(data_set, cate_features)


# Feature - "Electrical" has Missing values, So Mode is used to fill Nan
data_set['Electrical'].unique()
replace_null_values(data_set, ['Electrical'])
data_set['Electrical'].unique()

                               
# Replacing Nan -> with a new category : NA
null_replace_col = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
for col in cate_features :
    if data_set[col].isnull().any():
        data_set.loc[data_set[col].isnull(), col] = 'NO'
        
cat_null_percent = get_null_percentage(data_set, cate_features)


# Neighborhood Vs SalePrice
print(data_set['Neighborhood'].unique())
plt.figure(figsize=(30, 30))
plt.scatter(data_set['Neighborhood'], data_set['SalePrice'])

neb_condition_1 = (data_set['SalePrice'] >= 0) & (data_set['SalePrice'] < 200000)
neb_condition_2 = (data_set['SalePrice'] >= 200000) & (data_set['SalePrice'] < 400000)
neb_condition_3 = (data_set['SalePrice'] >= 400000) & (data_set['SalePrice'] < 600000)
neb_condition_4 = (data_set['SalePrice'] >= 600000)

print(data_set.loc[neb_condition_1, 'Neighborhood'].unique())
print(data_set.loc[neb_condition_2, 'Neighborhood'].unique())
print(data_set.loc[neb_condition_3, 'Neighborhood'].unique())
print(data_set.loc[neb_condition_4, 'Neighborhood'].unique())


data_set.loc[neb_condition_1, 'Neighborhood'] = '1'
data_set.loc[neb_condition_2, 'Neighborhood'] = '2'
data_set.loc[neb_condition_3, 'Neighborhood'] = '3'
data_set.loc[neb_condition_4, 'Neighborhood'] = '4'

# Update Features after cate_feat engineerring
classifiy_features(data_set, data_set.columns, year_features)

    
# Updating Numerical Features Using interpolate & Removing Outliers

# Updating Null values
num_null_percent = get_null_percentage(data_set, numeric_features)
data_set.interpolate(method='linear', inplace=True)
num_null_percent = get_null_percentage(data_set, numeric_features)

classifiy_features(data_set, data_set.columns, year_features)

# Removing Outliers
for col in numeric_features:
    if (col == 'LotFrontage'):
        get_outlier_details(data_set, col)
        data_set[col] = remove_outlier(data_set, col)

"""
    Choosing important Features :
    Method : SFS
"""

# Moving 'SalePrice' to last Pos
sale_price = data_set['SalePrice']
data_set.drop(['SalePrice'], axis=1, inplace=True)
data_set['SalePrice'] = sale_price


train_X = data_set.iloc[:, 0:76]
train_y = data_set.iloc[:, [76]].values

# Saving the Train Data
picklet_target_out = open('Housing-predictions-target-train.pkl', 'wb')
pickle.dump(train_y, picklet_target_out)
picklet_target_out.close()

# Final Column list before Dummies, remove columns from test data 
# if they are missed
data_set_cols = get_column_names(train_X.columns)

#num feat for flask :
numeric_features = []
cate_features = []
year_features = ['YearBuilt', 'YearRemodAdd']
classifiy_features(train_X, train_X.columns, year_features)

flk_num_feat = numeric_features

print(len(train_X.columns))


# Adding Dummies

# year dummies
year_dummy_data = pd.get_dummies(data_set[year_features], drop_first=True)


# cate dummies
cate_dummy_data = pd.get_dummies(data_set[cate_features], drop_first=True)


# Choosing best Categorical features:
temp_cate_feat = list(cate_dummy_data.columns)
cols = list(cate_dummy_data.columns)
train_target_var = train_y

pmax = 1
while (len(cols)>0):
    p= []
    X_train = cate_dummy_data[cols]
    X_train = sm.add_constant(X_train)
    model = sm.OLS(train_target_var, X_train).fit()
    p = pd.Series(model.pvalues.values[1:], index = cols)      
    pmax = max(p)
    print(pmax)
    feature_with_p_max = p.idxmax()
    print(feature_with_p_max)
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_cate_features = cols
print(selected_cate_features, '\n', len(selected_cate_features))

cate_dummy_columns = cate_dummy_data.columns
cate_dummy_columns_str = ''
for col in cate_dummy_columns:
    cate_dummy_columns_str += '\'' + col +'\','
        

cate_dummy_data = cate_dummy_data[selected_cate_features]


# Merging dummies
train_X = pd.concat([train_X, year_dummy_data, cate_dummy_data], axis = 1)
train_X.drop(year_features, axis=1, inplace=True)
train_X.drop(cate_features, axis=1, inplace=True)


# Scaling
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)


#  Predicition with Hyperparameter Tuning

# Hyperparameter Tuning
parameters = {
  "learning_rate"    : [0.05, 0.10, 0.15] ,
  "min_child_weight" : [ 1, 3, 5, 7 ],
  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
  "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
  "n_estimators"     : [300, 500]
}


regressor =  XGBRegressor(random_state=0)
grid_search = GridSearchCV(estimator = regressor, 
                            param_grid = parameters, scoring = 'r2', 
                            n_jobs = -1, cv = 10)
grid_search.fit(train_X, train_y)

"""
Best parameters:
    {'max_depth': 8, 'min_child_weight': 15, 'gamma' : 0.0, 
    'colsample_bylevel': 0.7, 'subsample': 0.7, 'reg_alpha': 0.01}
"""


train_predict = grid_search.predict(train_X)
r2 = r2_score(train_y, train_predict)
adj_r2 = 1 - ((1 - r2) * ((train_X.shape[0] - 1) / (train_X.shape[0] -
                                                    train_X.shape[1] - 1)))



# =============================================================================
# WORKING WITH TEST DATA
# =============================================================================

test_data_set = pd.read_csv('C:/Users/SRIKANDAN/DataScience/Project/HousingPricePrediction/test.csv', 
                            index_col = 'Id')
test_data_set.info()

test_data_shape = test_data_set.shape

''' 
Correlated features :  YearBuilt - GarageYrBlt, TotalBsmtSF - 1stFlrSF, 
                        TotRmsAbvGrd - GrLivArea, GarageCars - GarageArea,
So we are droping above correlated features 
''' 
corr_drop_cols = ['GarageYrBlt', 'GarageCars']
test_data_set.drop(corr_drop_cols, axis=1, inplace=True)


# Classifying Features
test_data_cols = get_column_names(test_data_set.columns)

test_numeric_features = []
test_cate_features = []
test_year_features = ['YearBuilt', 'YearRemodAdd']
classifiy_test_features(test_data_set, test_data_set.columns, test_year_features)


"""
    Updating DateTime Values
    Idead : Converting Year to Categorical Values with 5 Categories
    1   -> 1988 - 1987
    2   -> 1942 - 1941
    3   -> 1900 - 1918
    0   -> For Nan
"""
test_data_set = update_year_into_category(test_data_set, ['YearBuilt', 'YearRemodAdd'])
test_data_set.drop(['YrSold'], axis=1, inplace=True)


"""
    Updating Catrgorical Values
    Removing Features which has more than 85% missing values
    
    Features -    "MSZoning, Utilities, Exterior1st, 
                  Exterior2nd, KitchenQual, SaleType",
                  has Missing values, So Mode is used to fill Nan
"""
test_cat_null_percent = get_null_percentage(test_data_set, test_cate_features)

test_null_cols = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType']
replace_null_values(test_data_set, test_null_cols)

for col in test_cate_features :
    if test_data_set[col].isnull().any() and col != 'Utilities':
        test_data_set.loc[test_data_set[col].isnull(), col] = 'NO'
        
test_data_set.loc[test_data_set['Utilities'].isnull(), 'Utilities'] = 'AllPub'
        
test_cat_null_percent = get_null_percentage(test_data_set, test_cate_features)


# Neighborhood Vs SalePrice
neb_cond_1 = ['Veenker', 'Crawfor', 'Mitchel', 'OldTown', 'BrkSide', 'Sawyer',
        'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'CollgCr', 'Edwards',
        'Timber', 'Gilbert', 'Somerst', 'NWAmes', 'ClearCr', 'NPkVill',
        'Blmngtn', 'BrDale', 'SWISU', 'NridgHt', 'Blueste', 'StoneBr',
        'NoRidge']
neb_cond_2 = ['CollgCr', 'NoRidge', 'Somerst', 'NWAmes', 'NridgHt', 'NAmes',
        'Mitchel', 'Veenker', 'ClearCr', 'Crawfor', 'SawyerW', 'Gilbert',
        'Timber', 'StoneBr', 'Edwards', 'OldTown', 'BrkSide', 'Blmngtn',
        'SWISU']
neb_cond_3 = ['StoneBr', 'NridgHt', 'OldTown', 'NoRidge', 'Somerst', 'CollgCr']
neb_cond_4 = ['NoRidge', 'NridgHt']

test_data_set.loc[test_data_set['Neighborhood'].isin(neb_cond_1) , 'Neighborhood'] = '1'
test_data_set.loc[test_data_set['Neighborhood'].isin(neb_cond_2) , 'Neighborhood'] = '2'
test_data_set.loc[test_data_set['Neighborhood'].isin(neb_cond_3) , 'Neighborhood'] = '3'
test_data_set.loc[test_data_set['Neighborhood'].isin(neb_cond_4) , 'Neighborhood'] = '4'

# Update Features after cate_feat engineerring
classifiy_test_features(test_data_set, test_data_set.columns, test_year_features)


# Updating Numerical Features Using interpolate & Removing Outliers
# Updating Null values
num_null_percent = get_null_percentage(test_data_set, test_numeric_features)
test_data_set.interpolate(method='linear', inplace=True)
num_null_percent = get_null_percentage(test_data_set, test_numeric_features)

# Removing Outliers
for col in test_numeric_features:
    if (col == 'LotFrontage'):
        get_outlier_details(test_data_set, col)
        test_data_set[col] = remove_outlier(test_data_set, col)

# Update Features after cate_feat engineerring
classifiy_test_features(test_data_set, test_data_set.columns, test_year_features)

"""
    Choosing important Features :
    Method : SFS
"""

test_X = test_data_set.iloc[:, 0:76]
    
# Final Column list after Dummies, remove columns from test data if they are missed
test_data_cols = get_column_names(test_X.columns)

# Adding Dummies

# year dummies
test_year_dummy_data = pd.get_dummies(test_data_set[test_year_features], drop_first=True)


# cate dummies
test_cate_dummy_data = pd.get_dummies(test_data_set[test_cate_features], drop_first=True)

# Adding coulmns in test data that are missed here
# Finding missing cols in test and adding  it in test
for col in cate_dummy_data.columns:
    if col not in test_cate_dummy_data.columns:
        test_cate_dummy_data[col] = np.zeros(shape=(len(test_cate_dummy_data), 1))

# Choosed cate from train
test_cate_dummy_data = test_cate_dummy_data[selected_cate_features]


test_X = pd.concat([test_X, test_year_dummy_data, test_cate_dummy_data], axis = 1)
test_X.drop(test_year_features, axis=1, inplace=True)
test_X.drop(test_cate_features, axis=1, inplace=True)


# Scaling
test_scalar = StandardScaler()
test_X = test_scalar.fit_transform(test_X)


# Predicition
test_predict = regressor.predict(test_X)

predicted_data = pd.DataFrame()
predicted_data['Id'] = test_data_set.index
predicted_data['SalePrice'] = test_predict
predicted_data.to_csv('HousingPricePrediction.csv', index=False)
