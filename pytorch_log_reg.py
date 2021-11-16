# -*- coding: utf-8 -*-
"""pytorch_log_reg

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZbtnEY94qureiWvv9ySKuAgTn6wnL8mo

## Intro to pytorch

As a primer to next week's lesson's lectures on deep learning, we will be learning to use the python libary `pytorch`, which is a library for accelerated numerical computing. To provide context, lets break down the implementation of linear regression with gradient descent we worked on last class 

Note: to install pytorch, follow the instructions here. Its **HIGHLY** recommended to install with conda https://pytorch.org/get-started/locally/

In numpy, we work with arrays. In pytorch we work with tensors. Tensors *generally* behave the same as numpy arrays, but there are some differences when it comes to merging tensors which we won't get into for now.
"""

import numpy as np 
import torch
arr = np.random.random((3, 3))
arr

tns = torch.randn(3,3)
tns

"""Pytorch is useful for performing computations on a GPU, but for now we will be working on the CPU, as working on the CPU makes pytorch feel very similar to numpy. Next lab, we will cover working with GPUs

### What does pytorch do?

pytorch implements **auto-differentiation**, which is a computational technique for calculating the derivative for *any* mathematical function. This is very useful for any gradient-based optimization method because it allows use to calculate gradients with having to do any calculus by hand. Pytorch achieves this by tracking the computations associated with parameters. 

In the example below, lets say I have a simple regression model $y=X•\beta$, with sum of squared error loss: $loss = sum((y - (X•\beta))^2)


- use `torch.nn.Parameter` to signify to pytorch that these are the parameters of a function that we want to calculate the gradient for. 

- then instance an optimizer, and register the parameters of my model with which will update the weights in my parameters

- make a prediction with my model. Note that we are required to use matrix multiplication here.    

- calculate the loss based on the prediction and true input. 

- calculate the gradient based on the loss using the `.backward()` method 
- advance the optimizer based on 

"""

### randomly initialize weights 
class Model:
    weights = torch.nn.Parameter(torch.randn(10))
model = Model()
print(model.weights.shape)

### instance an optimizer

optimizer = torch.optim.SGD([model.weights], lr = 0.0001) ## I pass the parameters of my model to the optimizer to signify that that they need to be optimized 

## example data (X)
input_tns = torch.randn(100,10)
print(input_tns.shape)
## example response (Y)
true_values = torch.randn(100)
print(true_values.shape)
## prediction
output_tns = torch.matmul(input_tns, model.weights )
## loss calculation
loss = torch.sum((true_values-output_tns)**2)

## calling `.backward()` on a tensor calculate the gradient for all parameters a tensor has interacted with
grad = loss.backward() ##<- this calculates the gradient, but does not return anything!! it is stored implicitly based on the computation graph 
print("gradient", grad)

### this updates the weights that I passed when instancing the model.
optimizer.step()

"""By default, torch track all the computations associated with any object containing `torch.nn.Parameters` instances, so that we can compute the gradient. However, some times, we don't want to calulate the gradient, such as when we are predicting on heldout data. We can temporarily disable the tracking for the gradient using the `torch.no_grad()` context: """

with torch.no_grad():
    output_tns_no_grad  = torch.matmul(input_tns, model.weights )

print("no grad", output_tns_no_grad)
print("has grad", output_tns)

"""To give you a side by side comparison, here is the regression model from a couple labs ago """

class BatchSGDRegressor:
    def __init__(self, learning_rate, chunk_size, max_iter, validation_data=None, use_validation=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = chunk_size
        self.validation_data = validation_data
        self.weights = None
        self.training_loss = []
        self.validation_loss = []
        self.use_validation = use_validation
        
    def fit(self, X, y):
        ## ***X is a numpy array of training data**
        self.weights=np.zeros((X.shape[1], 1))
        for i in range(self.max_iter):
          
          ### Get a batch of data
          batch_idx = np.random.choice(X.shape[0], self.batch_size, replace=False)
          X_batch = X[batch_idx]
          y_batch = y[batch_idx]

          ### prediction
          prediction = np.dot(X_batch, self.weights)
          
          ### calculate the gradient
          gradient = np.dot(X_batch.T, (prediction - y)) / X_batch.shape[0]
          
          ### update the model via gradient descent 
          self.weights = self.weights - (self.learning_rate * gradient)
          loss = np.sum((self.predict(X_batch) - y_batch)**2)

"""one of the parts of pytorch that can make it challenging is that generally everything is structured as an object, and we generally write pytorch code through object oriented programming.

Object oriented programming is a programming model that centers around objects rather than functions and logic. This can be useful for large, complex and regularly updated programs that can be used for e.g., manufacturing and design, as well as mobile applications. Benefits of OOP include modularity, reusability, easily upgradable and scalable, flexibility and can help with security.
A large part of OOP is creating classes to organize these objects. A class is a user-defined data type that act as the blueprint for individual objects, attributes and methods. Where attributes represent the state of the class (basically variables within the class) and methods are essentially functions that are defined for the particular class.
Basic components of a class include the __init__ function which is used every time you create an instance of your class. It is used to define attributes that you want your object to have as soon as it's created. To understand a class vs. and object (which is an instantiation of a class) consider this example: you have a class "dog" and each dog has attributes such as hypoallergenic (y/n), length of fur/hair, energy level, size, etc. Classes can have funcitons associated with them, called methods, that can use both data stored within the class and take arguments for outside data.
Below is example code on creating a class:
```
class Dog:
   def __init__(size, energy_level, hypoallergenic, hair_length):
       self.size = size
       self.energy_level = energy_level
       self.hypoallergenic = hypoallergenic
       self.hair_length = hair_length
```

In this example I created a class Dog. When I create an instance of the class dog I would need to input the size, energy level, hypoallergenic status and hair length e.g.,
Rosie = Dog('medium', 'high', False, 'short')
You can then have specific methods for this class e.g.,

```
class Dog:
   def __init__(size, energy_level, hypoallergenic, hair_length, tired):
       self.size = size
       self.energy_level = energy_level
       self.hypoallergenic = hypoallergenic
       self.hair_length = hair_length
       self.tired = tired

   def walk(self, min):
       if min >= 30:
           self.tired = True
```
In this example, I added a new attribute to mark whether the dog is tired or not. I also added a method walk() which takes self and minutes and will update the attribute to tired = True if the dog is walked for more than 30 minutes otherwise that attribute will stay the same.

One of the biggest benefits to classes is that you can easily use code by using **inheritance** to hierachically structure code. This allowd code to by more re-usable.


"""

class Pet:
    def __init__(self, size, energy_level, hypoallergenic):
        self.size = size
        self.energy_level = energy_level
        self.hypoallergenic = hypoallergenic
    def walk(self):
        print("I am walking")
    def sleep(self):
        print("Zzz")

class Dog(Pet): ## dog is an instance of pet 
    def __init__(self, size, energy_level, hypoallergenic, toy_type):
        super(Dog, self).__init__(size, energy_level, hypoallergenic) ## calling super will initialize the parent class. this is allows you to access methods from the parent class
        self.toy_type = toy_type
    def __call__(self):
      self.fetch()
    def fetch(self):
        return f" go get the {self.toy_type}"


class Cat(Pet):
    def __init__(self, size, energy_level, hypoallergenic):
        super(Cat, self).__init__(size, energy_level, hypoallergenic)
    def meow(self):
        print("Meow")
        

dog = Dog(10,10,False, "ball")
dog.sleep()
print(dog.fetch())

cat = Cat(3, 4, True)
cat.walk()
cat.meow()



"""In pytorch most of the library code is available as a class, with the majority objects like models and loss functions based around the `torch.nn.Module` class. optimizers are based on a separate class To implement a new model or loss function, make sure to inherit `torch.nn.Module`  and instance it like so: Anything that implements `torch.nn.Module` must also implement the `forward method` The forward method is implements the action you'ld like the object to do when it's called. ie for regression its make a prediction, for mean squared error its calulcate the loss
```
class LinRegModel(torch.nn.Module):
    def __init__()
        super(MyModel, self).__init__()
        ...
    def forward(self, X)
        pred = torch.matmul(X, self.weights)
        return pred

class MeanSquaredError(torch.nn.Module):
    def __init__()
        super(MeanSquaredError, self).__init__()
    def forward(self, predicted, target)
        return torch.mean((predicted - target)**2)

X=<some data>
Y=<some data>
model = LinRegModel()
predicted = model(x) ## this is the same as doing model.forward(x)

```

# Task 4 (20 points)

You will be implementing logistic regression in pytorch. the formula for logistic regression is $\hat{y} = \sigma(X•\beta + \beta_0)$, where $\sigma$ is the sigmoid function

Note: pytorch can be finicky about datatypes for tensors. Its a good idea to make sure all input data is same type, ie `torch.float32`

- implement Logistic regression as a pytorch class, inheriting from `torch.nn.Module` and implementing the `forward` method. Use separate instances of `torch.nn.Parameter` for the weights term and bias term. intialize weights and bias randomly. Use matrix multiplication (ie `torch.matmul`) when implementing `forward`

- use binary cross entropy as your loss function

- split your data into training, validation, and test, using a 80/10/10 split and convert them to tensors

- Write you own function to sample batches from your training dataset, or use the [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) and [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)` classes in pytorch. If you implement your own, it should sample without replace and use all samples within the training dataset. 

- Using stochastic gradient descent(SGD) as your optimizer, write a loop that trains your model following this pseudocode. Make sure to call `.zero_grad()` on your optimizer at before passing data to the model.(This basically tells the optimizer that new data is coming)
```
model = MyModel()
optimizer = ... ## Hint: any class that inherits from 
 `torch.nn.Module` can access it's parameters via the `.parameters()` method
for e in n_epochs:
    for b in batches:
        optimizer.zero_grad()
        X,Y = get_batch
        ## get prediction
        ## calculate and store loss for training data
        ## calculate gradient
        ## use optimizer to update weights 
        ## Disable gradient tracking and calculate and store loss for validation data

```

- plot the validation and train loss as a function of batch 
- try out different values for the learning rate, pick the best one, and predict on the test data


Note that the major advantage of pytorch in this scenario is that we never have to explictly calculate what the gradient will be by hand( it is not [fun](https://blossom-quasar-68b.notion.site/Logistic-regression-2c57db55d9a548b982d18e00f72a4c1b))

### Load and prepare data
"""

# Library imports
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2023)

# Upload data
from google.colab import drive
drive.mount('/content/drive/')
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/AlQuraishi Lab/BRCA.csv')

# Drop these two columns because they had the same value for all entries
df.drop(['ER status', 'PR status'], axis = 1, inplace = True)
# Drop irrelevant data columns
df.drop(['Patient_ID', 'Date_of_Surgery', 'Date_of_Last_Visit'], axis = 1, inplace = True)
# Drop rows with N/As
df = df.dropna(axis=0, how='any') 
df.shape

# First, replace categorical features with two options with 0/1
df_replaced = df.replace({"FEMALE":0, "MALE":1, "Negative":0, "Positive":1, "Dead":0, "Alive":1})
# Next, replace categorical features with dummy variables
df_dummies = pd.get_dummies(df_replaced, columns = ['Tumour_Stage', 'Histology', 'Surgery_type'])
# Drop last dummy column for each categorical feature since it's redundant
df_dummies.drop(['Histology_Mucinous Carcinoma', 'Surgery_type_Other','Tumour_Stage_III'], axis = 1, inplace = True)

# Create X and y np arrays, specific to this example
def X_y_split(df):
  y_df = df['Protein1']
  X_df = df.drop(columns='Protein1')

  #Turn into np arrays
  y = np.array(y_df,'float')
  y = y.reshape(len(y),1)
  X = np.array(X_df,'float')

  return X, y


def X_y_split_new(df):
  y_df = df['Patient_Status']
  X_df = df.drop(columns='Patient_Status')

  #Turn into np arrays
  y = np.array(y_df,'float')
  y = y.reshape(len(y),1)
  X = np.array(X_df,'float')

  return X, y

#split into X and y np arrays
X, y = X_y_split_new(df_dummies)




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
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, 0.2)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, 0.5)



# in pytorch, could have used torch.utils.data.random_split

"""### Pytorch logReg model"""

class LogRegModel(torch.nn.Module):
  def __init__(self, num_inputs):
    super(LogRegModel, self).__init__()
    self.num_inputs = num_inputs
    self.weights = torch.nn.Parameter(torch.zeros(num_inputs, 1))
    self.bias = torch.nn.Parameter(torch.zeros(1))

  def forward(self, X):
    pred = torch.nn.Sigmoid()(torch.matmul(X, self.weights) + self.bias)
    
    return pred

"""### Get tensors"""

xtrain = torch.tensor(X_train) # transform to torch tensor
ytrain = torch.tensor(y_train)
xval = torch.Tensor(X_val)
yval = torch.Tensor(y_val)
xtest = torch.Tensor(X_test)
ytest = torch.Tensor(y_test)


batch_size=20

train_dataset = torch.utils.data.TensorDataset(xtrain,ytrain) # create datset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # create dataloader

"""
#these might be unncessary
val_dataset = torch.utils.data.TensorDataset(xval,yval)
val_dataloader = torch.utils.data.DataLoader(val_dataset) 

test_dataset = torch.utils.data.TensorDataset(xtest,ytest)
test_dataloader = torch.utils.data.DataLoader(test_dataset)
"""

"""### Training epochs"""

n_epochs = 25
lr = 0.00007

def_model = LogRegModel(14)
optimizer = torch.optim.SGD(def_model.parameters(), lr=lr)
bce_loss = torch.nn.BCELoss()

losses_train = []
losses_val = []

for e in range(n_epochs):
  for batch in train_dataloader:
      optimizer.zero_grad()
      x, y = batch
      pred = def_model.forward(x) ## get prediction
      loss_train = bce_loss(pred,y) ## calculate and store loss for training data
      losses_train.append(loss_train.item())
      grad = loss_train.backward() ## calculate gradient
      optimizer.step() ## use optimizer to update weights 

      with torch.no_grad(): ## Disable gradient tracking and calculate and store loss for validation data
        pred_val  = def_model.forward(xval)
        loss_val = bce_loss(pred_val,yval)
        losses_val.append(loss_val.item())

[p for p in def_model.parameters()]

plt.plot(losses_train)
plt.title("Train loss over batches")
plt.xlabel("Batch")
plt.ylabel("Train loss")

plt.plot(losses_val)
plt.title("Validation loss over batches")
plt.xlabel("Batch")
plt.ylabel("Validation loss")

print(losses_val[-5:-1])

"""### Predicting on test data"""

with torch.no_grad(): ## Disable gradient tracking and calculate and store loss for validation data
        pred_test  = model.forward(xtest)
        loss_test = bce_loss(pred_test,ytest)

print(loss_test)

"""# Task 5 (5 points)

Now, implement a custom loss function class which is an instance of `torch.nn.Module` adn implements the `.forward()` method, that adds in L1 and L2 regularization.  The loss function for this would look like $$BCELoss(y, \hat{y}) + \lambda_1 \sum_{j=0}^m |\beta_j| + \lambda_2 \sum_{j=0}^m \beta_j^2 $$ where $m$ is the number of features.

Retrain you model following the same procedure as above but using your custom loss function this time, and picking appropriate values for $\lambda_1$ and $\lambda_2$(Note: make sure that these are tensor scalars, not regular ones ie `lambda_1 = torch.tensor(1)`). Plot the distribution of the weights for the first model and this regularized one. 

"""

x = [torch.tensor(p.clone().detach().flatten()) for p in model.parameters()]



class RegularizationLoss(torch.nn.Module):
  def __init__(self, lambda_1, lambda_2):
      super(RegularizationLoss, self).__init__()
      self.lambda_1 = lambda_1
      self.lambda_2 = lambda_2
  def forward(self, predicted, target, model):
      x = [torch.tensor(p.clone().detach().flatten()) for p in model.parameters()]
      params  = torch.cat(x)
      #params = [torch.tensor(p.clone().detach()) for p in model.parameters()]
      regularization_loss = self.lambda_1 * torch.sum(torch.abs(params)) + self.lambda_2 * torch.sum(params**2)
      # for param in model.parameters():
      #   regularization_terms += self.lambda_1 * torch.sum(torch.abs(param)) + self.lambda_2 * torch.sum(param**2)
      loss = torch.nn.BCELoss()(predicted, target) + regularization_loss
      return loss

n_epochs = 30
lr = 0.0001

model = LogRegModel(14)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
reg_loss = RegularizationLoss(torch.tensor(20), torch.tensor(20))

losses_train = []
losses_val = []

for e in range(n_epochs):
  for batch in train_dataloader:
      optimizer.zero_grad()
      x, y = batch
      pred = model.forward(x) ## get prediction
      loss_train = reg_loss(pred,y, model) ## calculate and store loss for training data
      losses_train.append(loss_train.item())
      grad = loss_train.backward() ## calculate gradient
      optimizer.step() ## use optimizer to update weights 

      with torch.no_grad(): ## Disable gradient tracking and calculate and store loss for validation data
        pred_val  = model.forward(xval)
        loss_val = reg_loss(pred_val,yval, model)
        losses_val.append(loss_val.item())

plt.plot(losses_train)
plt.title("Train losses over batches")
plt.xlabel("Batch")
plt.ylabel("Train loss")

plt.plot(losses_val)
plt.title("Validation losses over batches")
plt.xlabel("Batch")
plt.ylabel("Validation loss")

[ p for p in model.parameters()]

