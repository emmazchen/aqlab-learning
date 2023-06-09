# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2023)

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

"""## X and y split"""

# Create X and y np arrays, specific to this example
def X_y_split(df):
  y_df = df['Protein1']
  X_df = df.drop(columns='Protein1')

  #Turn into np arrays
  y = np.array(y_df,'float')
  y = y.reshape(len(y),1)
  X = np.array(X_df,'float')

  return X, y

#split into X and y
X, y = X_y_split(df_dummies)

"""# Train test split"""

# Split into train and test set
def train_test_split(X, y, test_ratio=0.3):
  #concat X and y to shuffle together
  y_conc = np.atleast_2d(y)
  Xy = np.concatenate((X,y_conc), axis=1) 
  Xy_perm = np.random.permutation(Xy) #shuffle

  #split into train test based on ratio
  cutoff = int(test_ratio * len(Xy))
  Xy_test, Xy_train = Xy_perm[ :cutoff, :], Xy_perm[cutoff: , :]

  #split back into X and y
  X_train, y_train_list = np.array(Xy_train[:, :-1]), np.array(Xy_train[:, -1])
  X_test, y_test_list = np.array(Xy_test[:, :-1]), np.array(Xy_test[:, -1])
  y_train, y_test = y_train_list.reshape((len(y_train_list), 1)), y_test_list.reshape((len(y_test_list), 1))

  return X_train, X_test, y_train, y_test

#split twice into train, val, test sets
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, 0.3)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, 0.5)

X_st = (X - np.mean(X, 0)) / np.std(X, 0)

"""# Regression classes

To implement ridge regression, I added an alternative loss function.
"""

#Class that serves as a library of different loss functions to be used
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

#Optimization class with different regression functions that take in models and train them
                  #plot_loss function generates graph of loss over training iterations
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

# lambdas to try out, cv_nfold specifies how many fold
# uses leave one out strategy
#class CV():
#  def __init__(self, lambdas = [], cv_nfold = 4):

"""# CV"""

def CV (X, y, lambdas = [.0001, .001, .01, .1, 1], n_fold = 5):
  trainLosses = np.ndarray((len(lambdas), n_fold)) 
  valLosses = np.ndarray((len(lambdas), n_fold)) 

  for i in range(len(lambdas)):
    model = Model(14)
    training = Optimization(n_iter=20) 

    for j in range(n_fold): 
      start, end = int( j / n_fold * len(X)), int((j+1) / n_fold * len(X))
      Xt = np.concatenate((X[:start, :], X[end:, :]))
      Xv = np.array(X[start:end, :])
      yt = np.concatenate((y[:start, :], y[end:, :]))
      yv = np.array(y[start:end, :])

      training.ridge_regression(Xt, yt, model, ridge_lambda = lambdas[i])
      training.plot_loss() #Loss curve

      trainLosses[i][j] = Loss.mse(Xt, yt, model)
      valLosses[i][j] = Loss.mse(Xv, yv, model)

  avgTrainLoss = np.mean(trainLosses, axis = 1) #avg train loss for lambdas
  avgValLoss = np.mean(valLosses, axis = 1) #avg val loss for lambdas

  for i in range(len(lambdas)):
    print("Lambda: ", lambdas[i], ", avgValLoss = ", avgValLoss[i])

#test CV
X = X_train
y = y_train
lambdas = [.0001, .001, .01, .1, 1]
n_fold = 5

trainLosses = np.ndarray((len(lambdas), n_fold)) 
valLosses = np.ndarray((len(lambdas), n_fold)) 

for i in range(len(lambdas)):
  model = Model(14)
  training = Optimization(n_iter=20) 

  for j in range(n_fold): 
    start, end = int( j / n_fold * len(X)), int((j+1) / n_fold * len(X))
    Xt1, Xt2 = X[:start, :], X[end:, :]
    Xt = np.concatenate((Xt1, Xt2))
    Xv = np.array(X[start:end, :])
    yt1, yt2 = y[:start, :], y[end:, :]
    yt = np.concatenate((yt1, yt2))
    yv = np.array(y[start:end, :])

    training.ridge_regression(Xt, yt, model, ridge_lambda = lambdas[i])

    trainLosses[i][j] = Loss.mse(Xt, yt, model)
    valLosses[i][j] = Loss.mse(Xv, yv, model)

    if (i==1 and j==1):
      print(Xt.shape)
    #print(i, ", ", j, " trainLoss: ", trainLosses[i][j])
    #print(i, ", ", j, " valLoss: ", valLosses[i][j])

avgTrainLoss = np.mean(trainLosses, axis = 1) #avg train loss for lambdas
avgValLoss = np.mean(valLosses, axis = 1) #avg val loss for lambdas

for i in range(len(lambdas)):
  print("Lambda: ", lambdas[i], ", avgValLoss = ", avgValLoss[i])

X.shape

CV(X_train, y_train)

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
