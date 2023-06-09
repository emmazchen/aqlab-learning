"""
- In this version, I wrote separate classes for Loss, Model, and Optimization. <br>
- Also, models don't store  data internally but instead take data in each time method executes.
"""

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

"""# Linear regression classes"""

class Loss(): #static class

  def mse(X, y, model):
    n_samples = len(y)
    squared_errors = (y - model.predict(X)) ** 2
    mse = np.sum(squared_errors) / n_samples
    return mse

class Optimization():

  def __init__(self, alpha=0.03, n_iter=100):
      self.alpha = alpha
      self.n_iter = n_iter
      self.loss_history = np.zeros((n_iter,1))

  def fit(self, X, y, model):
      X_st = (X - np.mean(X, 0)) / np.std(X, 0) #standardize X
      y_st = (y - np.mean(y))/np.std(y) #standardize y
      model.intercept = np.mean(y) #set model intercept
      n_samples = np.size(X, 0)
      
      for i in range(self.n_iter):
          gradient = (1/n_samples) * X_st.T @ (X_st @ model.weights - y_st)
          model.weights = model.weights - self.alpha * gradient # update model weights
          self.loss_history[i] = Loss.mse(X, y, model) #store loss

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

  def get_weights(self):
        return self.weights

"""#Test run"""

X_train.shape #from this we know there are 14 features

#Declare model and training
model = Model(14)
training = Optimization(n_iter=20)

"""##Pre-training eval"""

#Pre-training losses
print("Pre-training \nLoss on train data: ", Loss.mse(X_train, y_train, model), "\nLoss on test data:", Loss.mse(X_test, y_test, model))

plt.scatter(y_train, model.predict(X_train), color='darkcyan')
plt.xlabel('Actual Protein1 level')
plt.ylabel('Predicted Protein1 level')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title('Actual vs Predicted Protein1 levels in train set')

plt.hist(y_train, histtype='step', stacked=True, fill=False, label='y (train set)', color="darkcyan")
plt.hist(model.predict(X_train), histtype='step', stacked=True, fill=False, label = 'y hat (train set)', color="springgreen")
plt.xlabel('Protein1 level')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Actual vs predicted Protein1 level distributions')

"""##Training"""

#Training with gradient descent
training.fit(X_train, y_train, model)
training.plot_loss() #Loss curve

"""## Post-training eval"""

#Post-training losses
print("Post-training \nLoss on train data: ", Loss.mse(X_train, y_train, model), "\nLoss on test data:", Loss.mse(X_test, y_test, model))

plt.scatter(y_train, model.predict(X_train), color='darkcyan')
plt.xlabel('Actual Protein1 level')
plt.ylabel('Predicted Protein1 level')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title('Actual vs Predicted Protein1 levels in train set')

plt.scatter(y_test, model.predict(X_test), color='brown')
plt.xlabel('Actual Protein1 level')
plt.ylabel('Predicted Protein1 level')
plt.xlim(-2, 1.5)
plt.ylim(-2, 1.5)
plt.title('Actual vs Predicted Protein1 levels in test set')

plt.hist(y_train, histtype='step', stacked=True, fill=False, label='y', color="darkcyan")
plt.hist(model.predict(X_train), histtype='step', stacked=True, fill=False, label = 'y hat', color="springgreen")
plt.xlabel('Protein1 level')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Actual vs predicted Protein1 level distributions (train set)')

plt.hist(y_test, histtype='step', stacked=True, fill=False, label='y', color="brown")
plt.hist(model.predict(X_test), histtype='step', stacked=True, fill=False, label = 'y hat', color="coral")
plt.xlabel('Protein1 level')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Actual vs predicted Protein1 level distributions (test set)')

# Define error metric for visualization
def error(y, y_hat):
  e = np.subtract(y_hat,y)
  return e

# Plot error metric on train and test data
plt.hist(error(y_train, model.predict(X_train)), color="darkcyan")
plt.title('Error in prediction for train set')
plt.xlabel('Error (y_hat - y)')
plt.ylabel('Number of patients')
plt.show()

plt.hist(error(y_test, model.predict(X_test)), color="brown")
plt.title('Error in prediction for test set')
plt.xlabel('Error (y_hat - y)')
plt.ylabel('Number of patients')
plt.show()

"""# Experiment: split data and two models"""

# Split into train and test set
df_A, df_B = train_test_split(df_dummies, ratio=2)

# Get X, y arrays for train and test sets
X_A, y_A = create_X_y(df_A)
X_B, y_B = create_X_y(df_B)


#Train two separate models
model_A = Model(14)
model_B = Model(14)


#Compare weights
training.fit(X_A, y_A, model_A)
training.fit(X_B, y_B, model_B)

#Post-training losses
print("Loss \nModel A on data A: ", Loss.mse(X_A, y_A, model_A), "\nModel A on data B: ", Loss.mse(X_B, y_B, model_A), "\nModel B on data B: ", Loss.mse(X_B, y_B, model_B), "\nModel B on data A: ", Loss.mse(X_A, y_A, model_B))

barWidth = 0.25

# set height of bar
bar1 = np.ndarray.tolist(model_A.weights.T)[0]
bar2 = np.ndarray.tolist(model_B.weights.T)[0]

 
# Set position of bar on X axis
br1 = np.arange(len(bar1))
br2 = [x + barWidth for x in br1]


plt.bar(br1, bar1, width = barWidth,
        edgecolor ='black', label ='model A weights')
plt.bar(br2, bar2, width = barWidth,
        edgecolor ='black', label ='model B weights')
plt.xticks([]) 
plt.xlabel("the 14 different weights")
plt.ylabel("value of weights")
plt.title("")
plt.legend()

"""It's interesting to see how the two models ended up with similar values for some weights but not others. <br>
My hypothesis is that the weights with high concordance are more relevant.

Roughly, <br>
High concordance: 2, 3, 5, 12, 14 <br>
Medium concordance: 4, 6 <br>
Low concordance: 1, 7, 8, 9, 10, 11, 13
"""

df_dummies.drop(['Protein1'], 1)

"""Roughly, <br>
High concordance: gender(2), protein2(3), protein4(5), surgerytype(12), surgerytype(14) <br>
Medium concordance: protein3(4), HER2 status(6) <br>
Low concordance: age (1), patientstatus (7), tumorstage I(8), tumorstage II(9), histology(10), histology(11), surgerytype(13)

Need to perform more trials!
"""
