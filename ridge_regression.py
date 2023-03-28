# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Upload data
from google.colab import drive
drive.mount('/content/drive/')

df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/AlQuraishi Lab/BRCA.csv')
df.shape # (341, 16)

"""# Preprocessing"""

# Drop these two columns because they had the same value for all entries
df.drop(['ER status', 'PR status'], axis = 1, inplace = True)

# Drop irrelevant data columns
df.drop(['Patient_ID', 'Date_of_Surgery', 'Date_of_Last_Visit'], axis = 1, inplace = True)

"""## Deal with missing data


"""

# Drop rows with all N/As
df = df.dropna(axis=0, how='all') 
df.shape

df.isna().sum()

# Decided to drop all rows with missing data since it's only a difference between 334 rows and 321 rows. >95% of rows preserved
df = df.dropna(axis=0, how='any') 
df.shape

"""## Encode categorical features"""

df.value_counts('Histology') # to see different values discrete var can take on

# features to be dealt with:  
#     Gender (2), ER status (2), PR status (2), HER2 status (2), Patient_Status (2),  -> directly use 0/1
#     Histology (3), Surgery type (4), Tumour_Stage (3), -> indicator/dummy variables
#     patient ID, Date_of_Surgery (date), Date_of_Last_Visit (date) -> drop because not relevant

# First, replace categorical features with two options with 0/1
df_replaced = df.replace({"FEMALE":0, "MALE":1, "Negative":0, "Positive":1, "Dead":0, "Alive":1})

# Next, replace categorical features with dummy variables
df_dummies = pd.get_dummies(df_replaced, columns = ['Tumour_Stage', 'Histology', 'Surgery_type'])

# Drop last dummy column for each categorical feature since it's redundant
df_dummies.drop(['Histology_Mucinous Carcinoma', 'Surgery_type_Other','Tumour_Stage_III'], axis = 1, inplace = True)

df_dummies

df_dummies.shape

"""## Prepare train and test set"""

import random

# Split into train and test set
def train_test_split(df, ratio=5, n=0, randomize=False):
  # Default is ratio = 5, meaning 1/5 = 20% becomes the test set
  # This enables us to choose different subsets to be the test set
  if randomize == True:
    n = random.randint(0,4)

  df_test = df[(df.index+n) % ratio == 0]
  df_train = df[(df.index+n) % ratio != 0]
  return df_train, df_test


# Create X and y np arrays
def create_X_y(df):
  y_df = df['Protein1']
  X_df = df.drop(columns='Protein1')

  #Turn into np arrays
  y = np.array(y_df,'float')
  y = y.reshape(len(y),1)
  X = np.array(X_df,'float')

  return X, y

# Split into train and test set
df_train, df_test = train_test_split(df_dummies)

# Get X, y arrays for train and test sets
X_train, y_train = create_X_y(df_train)
X_test, y_test = create_X_y(df_test)

"""# Regression classes

To implement ridge regression, I added an alternative loss function.
"""

class Loss(): #static class

  def mse(X, y, model):
    n_samples = len(y)
    squared_errors = (y - model.predict(X)) ** 2
    mse = np.sum(squared_errors) / n_samples
    return mse

  def sse(X, y, model): #wrote this as just another loss function option
    n_samples = len(y)
    squared_errors = (y - model.predict(X)) ** 2
    sse = np.sum(squared_errors)
    return sse

  def ridge_mse(X, y, model, lamda = 0.0001):
    mse = Loss.mse(X, y, model)
    penalty = lamda * (model.weights.T @ model.weights)
    ridge_mse = mse + penalty
    return ridge_mse

class Optimization():

  def __init__(self, alpha=0.03, n_iter=100):
      self.alpha = alpha
      self.n_iter = n_iter
      self.loss_history = np.zeros((n_iter,1))

  def linear_regression(self, X, y, model):
      X_st = (X - np.mean(X, 0)) / np.std(X, 0) #standardize X
      y_st = (y - np.mean(y))/np.std(y) #standardize y
      model.intercept = np.mean(y) #set model intercept
      n_samples = np.size(X, 0)
      for i in range(self.n_iter):
          gradient = (1/n_samples) * X_st.T @ (X_st @ model.weights - y_st) # gradient calc specific to lin reg
          model.weights = model.weights - self.alpha * gradient # update model weights
          self.loss_history[i] = Loss.mse(X, y, model) #store loss
      return self
  
  def ridge_regression(self, X, y, model, ridge_lambda = 0.0001): #different gradient calc and different loss_history
      X_st = (X - np.mean(X, 0)) / np.std(X, 0) 
      y_st = (y - np.mean(y))/np.std(y)
      model.intercept = np.mean(y)
      n_samples = np.size(X, 0)
      for i in range(self.n_iter):
          gradient = (1/n_samples) * X_st.T @ (X_st @ model.weights - y_st) + ridge_lambda * model.weights
          model.weights = model.weights - self.alpha * gradient 
          self.loss_history[i] = Loss.ridge_mse(X, y, model, ridge_lambda) 
      return self
  
  def plot_loss(self): #plot loss over iterations
    plt.plot(range(self.n_iter), self.loss_history)
    plt.title('Loss over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

class Model(): #training data is not part of model, only variables are n_features, weights, and intercept
  def __init__(self, n_features):  
    self.n_features = n_features
    self.weights = np.zeros((self.n_features, 1))
    self.intercept = 0

  def predict(self, X):  #predict on external supplied X
        n_samples = np.size(X, 0)
        X_st = (X - np.mean(X, 0)) / np.std(X, 0) #standardize X
        pred = X_st @ self.weights + self.intercept
        return pred

  def get_weights(self): #realized don't need this cuz python
        return self.weights

"""#Test run comparing linear and ridge regression"""

X_train.shape #from this we know there are 14 features

#Declare model and training
model_l = Model(14)
model_r = Model(14)
training_l = Optimization(n_iter=20)
training_r = Optimization(n_iter=20)

#Training w lin reg
training_l.linear_regression(X_train, y_train, model_l)
training_l.plot_loss() #Loss curve

#Training w ridge reg
training_r.ridge_regression(X_train, y_train, model_r, ridge_lambda=1)
training_r.plot_loss() #Loss curve

"""##Post-training eval"""

#Post-training mse for lin reg
print("Lin regression \nLoss on train data: ", Loss.mse(X_train, y_train, model_l), "\nLoss on test data:", Loss.mse(X_test, y_test, model_l))

#Post-training mse for ridge reg
print("Ridge regression \nLoss on train data: ", Loss.mse(X_train, y_train, model_r), "\nLoss on test data:", Loss.mse(X_test, y_test, model_r))

"""**Compare weights obtained**"""

barWidth = 0.25

# set height of bar
bar1 = np.ndarray.tolist(model_l.weights.T)[0]
bar2 = np.ndarray.tolist(model_r.weights.T)[0]

 
# Set position of bar on X axis
br1 = np.arange(len(bar1))
br2 = [x + barWidth for x in br1]


plt.bar(br1, bar1, width = barWidth,
        edgecolor ='black', label ='linear regression weights')
plt.bar(br2, bar2, width = barWidth,
        edgecolor ='black', label ='ridge regression weights')
plt.xticks([]) 
plt.xlabel("the 14 different weights")
plt.ylabel("value of weights")
plt.title("")
plt.legend()
