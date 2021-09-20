#!/usr/bin/env python
# coding: utf-8

# # DIT821 Software Engineering for AI systems
# 
# DIT821 labs are derived from excercises from Coursera Machine Learning course. 
# Two students cam work together, and individaully submit the solutions to CANVAS
# 
# <div class="alert alert-block alert-warning">
# You are supposed to solve them yourself and submit the solutions. Further, you will be asked individually to explan the labs, in particuar the parts that you have written. The labs will be approved upon sucssesful correct submission and discussion. 
# </div>

# Enter here your first and last name and your e-mail adress as regiistered in Canvas.
# 
# * Name, e-mail:
# * Name, e-mail:
# 

# ## Lab 2: Linear Regression 
# 
# ## Introduction
# 
# In this exercise, you will implement linear regression and get to see it work on data. Before starting on this programming exercise, we strongly recommend revisiting lecture slides.
# 
# All the information you need for solving this assignment is in this notebook, and all the code you will be implementing will take place within this notebook. The assignment should be submitted to Canvas(instructions are included below).
# 
# Before we begin with the exercises, we need to import all libraries required for this programming exercise. Throughout the course, we will be using:
# * [`numpy`](http://www.numpy.org/) for all arrays and matrix operations
# * [`matplotlib`](https://matplotlib.org/) for plotting.
# 
# You can find instructions on how to install required libraries in the README file in the [github repository] (https://github.com/pierg/ai-course).

# ## Exercise Structure
# 
# For this programming exercise, you are required to implement three parts of linear regression. The following is a breakdown of how each part of this exercise.
# * Part 1: Linear regression (one feature)
# * Part 2: Linear regression (multiple features)
# * Part 3: Linear regression (multiple features, Normal Equation)
# 
# Before begining the three parts, there is a simple exercise to help you familiarize with numpy in Python.
# 
# 
# | Part | Exercise                                           |Submitted Function                     | 
# |---------|:-                                             |:-                                     |     
# | 0       | [Warm up exercise](#section1)                  | [`warmUpExercise`](#warmUpExercise)    |   
# | 1       | [Compute cost for one feature](#section2)     | [`computeCost`](#computeCost)         |    
# | 1       | [Gradient descent for one feature](#section3) | [`gradientDescent`](#gradientDescent) |         
# | 2       | [Feature normalization](#section4)                   | [`featureNormalize`](#featureNormalize) |     |
# | 2       | [Compute cost for multiple features](#section5)     | [`computeCostMulti`](#computeCostMulti) | 0      |
# | 2       | [Gradient descent for multiple features](#section5) | [`gradientDescentMulti`](#gradientDescentMulti) |0      |
# | 3       | [Normal Equation](#section7)                        | [`normalEqn`](#normalEqn)        | 0      |
# 
# <!-- 
# You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration. -->
# 
# <!--
# <div class="alert alert-block alert-warning">
# At the end of each section in this notebook, we have a cell which contains code for submitting the solutions thus far to the grader. Execute the cell to see your score up to the current section. For all your work to be submitted properly, you must execute those cells at least once. They must also be re-executed everytime the submitted function is updated.
# </div> 
# -->
# 
# 
# ## Debugging
# 
# Here are some things to keep in mind throughout this exercise:
# 
# - Python array indices start from zero, not one (contrary to OCTAVE/MATLAB). 
# 
# - There is an important distinction between python arrays (called `list` or `tuple`) and `numpy` arrays. You should use `numpy` arrays in all your computations. Vector/matrix operations work only with `numpy` arrays. Python lists do not support vector operations (you need to use for loops).
# 
# - If you are seeing many errors at runtime, inspect your matrix operations to make sure that you are adding and multiplying matrices of compatible dimensions. Printing the dimensions of `numpy` arrays using the `shape` property will help you debug.
# 
# - By default, `numpy` interprets math operators to be element-wise operators. If you want to do matrix multiplication, you need to use the `dot` function in `numpy`. For, example if `A` and `B` are two `numpy` matrices, then the matrix operation AB is `np.dot(A, B)`. Note that for 2-dimensional matrices or vectors (1-dimensional), this is also equivalent to `A@B` (requires python >= 3.5).

# ## Exercise start
# 
# To be able to run the excercises you should run the part below (in Jupyter Noetbook called "a cell") to import libraries you will need.
# Note: Execute the cell either by klicling Run button or pressing Shift-Enter

# In[1]:


# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# library written for this exercise providing additional functions for assignment submission, and others
#import utils 

# define the submission/grader object for this exercise
#grader = utils.Grader()

# tells matplotlib to embed plots within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ##  Warm-up exercise 
# 
# This war-up gives you practice with python and `numpy` and `matplot`syntax. In the next cell, you will find the outline of a `python` function. Modify it to return a 5 x 5 identity matrix by filling in the following code:
# 
# ```python
# A = np.eye(5)
# ```

# In[2]:


def warmUpExercise():
    """
    Example function in Python which computes the identity matrix.
    
    Returns
    -------
    A : array_like
        The 5x5 identity matrix.
    
    Instructions
    ------------
    Return the 5x5 identity matrix.
    """    
    # ======== YOUR CODE HERE ======
     # modify this line
    A = np.eye(5)
    # ==============================
    return A


# <a id="section1"></a>
# ## Simple python: `numpy`  and `matplot` function
# 
# This warm-up exercise gives you practice with python and `numpy` and `matplot`syntax. 
# 
# <div class="alert alert-block alert-warning">
#     
# * A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers.
# * You initialize numpy arrays from nested Python lists, and access elements using square brackets
# ```python
#       a = np.array([1, 2, 3]) # Creates an array of 1 axis
# ```
# * NumPy’s array class is called ndarray (alias array) has the following important attributes (Note 'a' is the initialized array above): 
# ```python
#       a.shape # The dimension of array
#       a.size # Total number of elements of the array
#       a.dtype # Describes the type of the elements
# ```
# * Numpy also provides many functions to create arrays:
# ```python
#       a = np.zeros((2,2)) # Creates an array of all zeros
#       b = np.ones((1,2))  # Creates an array of all ones
#       c = np.full((2,2), 7)# Creates an array of constants
#       d = np.eye(2) # Creates a 2x2 identity matrix
#       e = np.random.random((2,2)) # Creates an array filled with random values
#  ```   
# </div>
# <a id="warmUpExercise"></a>

# The previous cell only defines the function `warmUpExercise`. We can now run it by executing the following cell to see its output. You should see output similar to the following:
# 
# ```python
# array([[ 1.,  0.,  0.,  0.,  0.],
#        [ 0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  0.,  1.]])
# ```

# In[3]:


warmUpExercise()


# ### Ploting graphs
# Before starting on any task, it is often useful to understand the data by visualizing it. For example, you can use a scatter plot to visualize the data provided in this exercise (Data/ex1data1.txt), since it has only two properties to plot (profit and population).
# 
# In this course, we will be exclusively using `matplotlib` to do all our plotting. `matplotlib` is one of the most popular scientific plotting libraries in python and has extensive tools and functions to make beautiful plots. `pyplot` is a module within `matplotlib` which provides a simplified interface to `matplotlib`'s most common plotting tasks.
# 
# In the following part, your job is to complete the `plotData` function below. Modify the function and fill in the following code:
# 
# ```python
#     pyplot.plot(x, y, 'ro', ms=10, mec='k')
#     pyplot.ylabel('Profit in $10,000')
#     pyplot.xlabel('Population of City in 10,000s')
#     pyplot.show()
# ```

# In[4]:


def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data 
    points and gives the figure axes labels of population and profit.
    
    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.
    
    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.    
    
    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You 
    can also set the marker edge color using the `mec` property.
    """
    fig = pyplot.figure()  # open a new figure
    
    # ====================== YOUR CODE HERE ======================= 
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in $10,000')
    pyplot.xlabel('Population of City in 10,000s')
    pyplot.show()
   
    # =============================================================


# You will use this plotData function below after loading the dataset. To quickly learn more about the `matplotlib` plot function and what arguments you can provide to it, you can type `?pyplot.plot` in a cell within the jupyter notebook. This opens a separate page showing the documentation for the requested function. You can also search online for plotting documentation. 
# 
# To set the markers to red circles, we used the option `'or'` within the `plot` function.

# In[5]:


get_ipython().run_line_magic('pinfo', 'pyplot.plot')


# ## PART 1: Linear Regression (one feature)
# 
# In this first part of the exercise you will implement linear regression with one feature to predict profits for a food truck. 
# 
# Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities. You would like to use this data to help you select which city to expand to next. 
# 
# The file `Data/ex1data1.txt` contains the dataset for our linear regression problem. The first column is the population of a city (in 10,000s) and the second column is the profit of a food truck in that city (in $10,000s). A negative value for profit indicates a loss. 
# 
# The below code is provided to load this data. The dataset is loaded from the data file into the variables `x` and `y`.
# 
# <div class="alert alert-block alert-warning">
# 
# * You can load data from text file using the following, however note that each row in the text file must have the same number of values.):
# ```python
#         np.loadtxt(fname, delimiter=None) #Parameter fname: file, str, or pathlib.Path
# ```
# * Numpy offers ways to index into arrays e.g., using slicing. Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array
# 
# ```python
# 
#        # Create an array with shape (3, 4)
#        a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# 
#        # Use slicing to pull out a new subarray 'b' consisting of the first 2 rows
#        # and columns 1 and 2; shape of b is (2, 2). Notice indexing starts at 0
#        # [[2 3]
#        #  [6 7]]
#        b = a[:2, 1:3]
# ```
# 
# </div>

# In[6]:


# Read comma separated data
data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]

m = y.size  # number of training examples

# if you wish to print the values of the variables uncomment the lines below
#print('X=\n',X)
#print('y=\n', y)
#print('m=', m)


# You can visualize the loaded data by running the `plotData` function defined earlier. Execute the next cell to visualize the data.

# In[7]:


plotData(X, y)


# ### Recall the following.
# 
# A linear function:
# $$ h(x) = \hat y = mx + b$$
# 
# Loss function is used to measure how much a data point is distant from the hypothesis. 
# 
# Square loss:
# $$ (y_i - \hat y_i)^2 $$
# 
# Cost function - loss function across the entire training set.
# 
# Summed Squared Loss (SSE):
# $$ \sum_{i=1}^N (y_i - \hat y_i)^2 $$
# 
# Mean Squared Error (MSE):
# $$ \frac{1}{N}\sum_{i=1}^N(y_i - \hat y_i)^2$$
# 
# And the sum in MSE is:
# $$ (y_{1}-(mx_1 + b))^2 + (y_{2}-(mx_2 + b))^2 + ...+ (y_{N}-(mx_N + b))^2 $$
# 
# 		
# The objective to minimize cost function with respect to $m$ and $b$:		
# $$ J(m, b) = (y_{1}-(mx_1 + b))^2 + (y_{2}-(mx_2 + b))^2 + ...+ (y_{N}-(mx_N + b))^2 $$
# 	
# 
# We redefine $m$ and $b$ with $w_1$ and $w_0$ respectively, cost function:
# $$ Loss(h_w) = \sum_{j=1}^N L_2(y_j, h_w(x_j)) = \sum_{j=1}^N (y_j- h_w(x_j))^2 = \sum_{j=1}^N(y_j - (w_1x_j+w_0))^2 $$
# 
# 
# where the hypothesis $h_w(x)$ is given by the linear model
# $$ h_w(x) = w_0 + w_1 x_1 + w_2x_2 + ....+ w_nx_n = w^T x$$
# 
# Thus, cost function using MSE:
# 
# $$ J(w) = \frac{1}{2m} \sum_{i=1}^m \left( h_{w}(x^{(i)}) - y^{(i)}\right)^2$$
# 
# 

# 
# Recall that the parameters of your model are the $w_j$ values. These are
# the values you will adjust to minimize cost $J(w)$. One way to do this is to
# use the batch gradient descent algorithm. In batch gradient descent, each
# iteration performs the update
# 
# $$ w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_w(x^{(i)}) - y^{(i)}\right)x_j^{(i)} \qquad \text{simultaneously update } w_j \text{ for all } j$$
# 
# $\alpha$ is a learning rate - a factor that defines the size of each step in the iterations.  
# 
# With each step of gradient descent, your parameters $w_j$ come closer to the optimal values that will achieve the lowest cost J($w$).
# 
# <div class="alert alert-block alert-warning">
# **Implementation Note:** We store each example as a row in the the $X$ matrix in Python `numpy`. To take into account the intercept term ($w_0$), we add an additional first column to $X$ and set it to all ones. This allows us to treat $w_0$ as simply another 'feature'.
# </div>
# 
# 
# #### Implementation
# 
# We have already set up the data for linear regression. In the following cell, we add another dimension to our data to accommodate the $w_0$ intercept term. Do NOT execute this cell more than once.

# In[8]:


# Add a column of ones to X. The numpy function stack joins arrays along a given axis. 
# The first axis (axis=0) refers to rows (training examples) 
# and second axis (axis=1) refers to columns (features).
X = np.stack([np.ones(m), X], axis=1)

# you can print the variable to see the result (remove comment character # and execute the cell)
# print(X) 


# <a id="section2"></a>
# #### 1.1 Computing the cost $J(w)$
# 
# As you perform gradient descent to learn minimize the cost function $J(w)$, it is helpful to monitor the convergence by computing the cost. In this section, you will implement a function to calculate $J(w)$ so you can check the convergence of your gradient descent implementation. 
# 
# Your next task is to complete the code for the function `computeCost` which computes $J(w)$. As you are doing this, remember that the variables $X$ and $y$ are not scalar values. $X$ is a matrix whose rows represent the examples from the training set and $y$ is a vector whose each element represent the value at a given row of $X$.
# <a id="computeCost"></a>
# 
# <div class="alert alert-box alert-warning">
# **Vectors and matrices in `numpy`** - Important implementation notes
# 
# A vector in `numpy` is a one dimensional array, for example `np.array([1, 2, 3])` is a vector. A matrix in `numpy` is a two dimensional array, for example `np.array([[1, 2, 3], [4, 5, 6]])`. However, the following is still considered a matrix `np.array([[1, 2, 3]])` since it has two dimensions, even if it has a shape of 1x3 (which looks like a vector).
# 
# Given the above, the function `np.dot` which we will use for all matrix/vector multiplication has the following properties:
# - It always performs inner products on vectors. If `x=np.array([1, 2, 3])`, then `np.dot(x, x)` is a scalar.
# - For matrix-vector multiplication, so if $X$ is a $m\times n$ matrix and $y$ is a vector of length $m$, then the operation `np.dot(y, X)` considers $y$ as a $1 \times m$ vector. On the other hand, if $y$ is a vector of length $n$, then the operation `np.dot(X, y)` considers $y$ as a $n \times 1$ matrix.
# - A vector can be promoted to a matrix using `y[None]` or `[y[np.newaxis]`. That is, if `y = np.array([1, 2, 3])` is a vector of size 3, then `y[None, :]` is a matrix of shape $1 \times 3$. We can use `y[:, None]` to obtain a shape of $3 \times 1$.
# <div>
# <a id="gradientDescent"></a>

# In[9]:


def computeCost(X, y, w):
    """
    Compute cost for linear regression. Computes the cost of using w as the
    parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m, n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of "1" already 
        appended to the features so we have n+1 columns.
    
    y : array_like
        The values of the function at each data point. This is a vector of
        dimension (m, ).
    
    w : array_like
        The parameters for the regression function. This is a vector of 
        dimension (n+1, ).
    
    Returns
    -------
    J : float
        The value of the regression cost function.
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. 
    You should set J to the cost.
    """
    
    # initialize some useful values
    m = y.size  # number of training examples
    
    # You need to return the following variables correctly
    J = 0
    
    # ====================== YOUR CODE HERE =====================
     # h=np.dot(X,w)
    J = sum((np.dot(X,w)-y)**2)/(2*m)
    # ===========================================================
    return J


# Once you have completed the function, the next step will run `computeCost` two times using two different initializations of $w$. You will see the cost printed to the screen.

# In[10]:


J = computeCost(X, y, w=np.array([0.0, 0.0]))
print('With w = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, w=np.array([-1, 2]))
print('With w = [-1, 2]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 54.24')


# <a id="section3"></a>
# #### 1.2. Gradient descent
# 
# Next, you will complete a function which implements gradient descent.
# The loop structure has been written for you, and you only need to supply the updates to $w$ within each iteration. 
# 
# As you program, make sure you understand what you are trying to optimize and what is being updated. Keep in mind that the cost $J(w)$ is parameterized by the vector $w$, not $X$ and $y$. That is, we minimize the value of $J(w)$ by changing the values of the vector $w$, not by changing $X$ or $y$. A good way to verify that gradient descent is working correctly is to look at the value of $J(w)$ and check that it is decreasing with each step. 
# 
# The starter code for the function `gradientDescent` calls `computeCost` on every iteration and saves the cost to a `python` list. Assuming you have implemented gradient descent and `computeCost` correctly, your value of $J(w)$ should never increase, and should converge to a steady value by the end of the algorithm.
# 

# In[11]:


def gradientDescent(X, y, w, alpha, num_iters):
    """
    Performs gradient descent to learn `w`. Updates w by taking `num_iters`
    gradient steps with learning rate `alpha`.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).
    
    y : arra_like
        Value at given features. A vector of shape (m, ).
    
    w : array_like
        Initial values for the linear regression parameters. 
        A vector of shape (n+1, ).
    
    alpha : float
        The learning rate.
    
    num_iters : int
        The number of iterations for gradient descent. 
    
    Returns
    -------
    w : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector w.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    #m = y.size
    # make a copy of w, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    w = w.copy()
 
    J_history = [] # Use a python list to save cost in every iteration
    
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        w = w - alpha/m*(np.dot(X,w)-y)@X
        # =====================================================================

        # save the cost J in every iteration
        J_history.append(computeCost(X, y, w))
    
    return w, J_history


# After you are finished call the implemented `gradientDescent` function and print the computed $w$. We initialize the $w$ parameters to 0 and the learning rate $\alpha$ to 0.01. Execute the following cell to check your code.

# In[12]:


# initialize fitting parameters
w = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

w, J_history = gradientDescent(X ,y, w, alpha, iterations)
print('W found by gradient descent: {:.4f}, {:.4f}'.format(*w))
print('Expected w values (approximately): [-3.6303, 1.1664]')


# We will use your final parameters to plot the linear fit. The results should look like the following figure.
# 
# ![](Figures/regression_result.png)

# In[13]:


# plot the linear fit
pyplot.plot(X[:,1], y,'ro', ms=10, mec= 'k')
#plotData(X[:, 1], y)
pyplot.plot(X[:, 1], np.dot(X, w), '-')
pyplot.legend(['Training data', 'Linear regression']);


# Your final values for $w$ will also be used to make predictions on profits in areas of 35,000 and 70,000 people.
# 
# <div class="alert alert-block alert-success">
# Note the way that the following lines use matrix multiplication, rather than explicit summation or looping, to calculate the predictions. This is an example of code vectorization in `numpy`.
# </div>
# 
# <div class="alert alert-block alert-success">
# Note that the first argument to the `numpy` function `dot` is a python list. `numpy` can internally converts **valid** python lists to numpy arrays when explicitly provided as arguments to `numpy` functions.
# </div>
# 

# In[14]:


# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], w)
print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))

predict2 = np.dot([1, 7], w)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))


# ###  Visualizing $J(w)$
# 
# To understand the cost function $J(w)$ better, you will now plot the cost over a 2-dimensional grid of $w_0$ and $w_1$ values. You will not need to code anything new for this part, but you should understand how the code you have written already is creating these images.
# 
# In the next cell, the code is set up to calculate $J(w)$ over a grid of values using the `computeCost` function that you wrote. After executing the following cell, you will have a 2-D array of $J(w)$ values. Then, those values are used to produce surface and contour plots of $J(w)$ using the matplotlib `plot_surface` and `contourf` functions. The plots should look something like the following:
# 
# ![](Figures/cost_function.png)
# 
# The purpose of these graphs is to show you how $J(w)$ varies with changes in $w_0$ and $w_1$. The cost function $J(w)$ is bowl-shaped and has a global minimum. (This is easier to see in the contour plot than in the 3D surface plot). This minimum is the optimal point for $w_0$ and $w_1$, and each step of gradient descent moves closer to this point.

# In[15]:


# grid over which we will calculate J
w0_vals = np.linspace(-10, 10, 100)
w1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((w0_vals.shape[0], w1_vals.shape[0]))

# Fill out J_vals
for i, w0 in enumerate(w0_vals):
    for j, w1 in enumerate(w1_vals):
        J_vals[i, j] = computeCost(X, y, [w0, w1])
        
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# surface plot
fig = pyplot.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(w0_vals, w1_vals, J_vals, cmap='viridis')
pyplot.xlabel('w0')
pyplot.ylabel('w1')
pyplot.title('Surface')

# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = pyplot.subplot(122)
pyplot.contour(w0_vals, w1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('w0')
pyplot.ylabel('w1')
pyplot.plot(w[0], w[1], 'ro', ms=10, lw=2)
pyplot.title('Contour, showing minimum')
pass


# ## PART 2: Linear regression (multiple features)
# 
# In this part, you will implement linear regression with multiple variables to predict the prices of houses. Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing prices.
# 
# The file `Data/ex1data2.txt` contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price
# of the house. 
# 
# <a id="section4"></a>
# ### 2.1 Feature Normalization
# 
# We start by loading and displaying some values from this dataset. By looking at the values, note that house sizes are about 1000 times the number of bedrooms. When features differ by orders of magnitude, first performing feature scaling can make gradient descent converge much more quickly.

# In[16]:


# Load data
data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))


# Your task here is to complete the code in `featureNormalize` function:
# - Subtract the mean value of each feature from the dataset.
# - After subtracting the mean, additionally scale (divide) the feature values by their respective “standard deviations.”
# 
# The standard deviation is a way of measuring how much variation there is in the range of values of a particular feature (most data points will lie within ±2 standard deviations of the mean); this is an alternative to taking the range of values (max-min). In `numpy`, you can use the `std` function to compute the standard deviation. 
# 
# For example, the quantity `X[:, 0]` contains all the values of $x_1$ (house sizes) in the training set, so `np.std(X[:, 0])` computes the standard deviation of the house sizes.
# At the time that the function `featureNormalize` is called, the extra column of 1’s corresponding to $x_0 = 1$ has not yet been added to $X$. 
# 
# You will do this for all the features and your code should work with datasets of all sizes (any number of features / examples). Note that each column of the matrix $X$ corresponds to one feature.
# 
# <div class="alert alert-block alert-warning">
# **Implementation Note:** When normalizing the features, it is important
# to store the values used for normalization - the mean value and the standard deviation used for the computations. After learning the parameters
# from the model, we often want to predict the prices of houses we have not
# seen before. Given a new x value (living room area and number of bedrooms), we must first normalize x using the mean and standard deviation that we had previously computed from the training set.
# </div>
# <a id="featureNormalize"></a>

# In[17]:


def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Note that X is a matrix where each column is a feature and each row is
    an example. You need to perform the normalization separately for each feature. 
    
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # =========================== YOUR CODE HERE =====================
    for idx in range(X_norm.shape[1]):
        mu[idx] = np.mean(X_norm[:,idx])
        sigma[idx] = np.std(X_norm[:,idx])
        tmu = np.array([mu[idx]]*X_norm.shape[0])
        X_norm[:,idx] = (X_norm[:,idx] - tmu)/sigma[idx]
    # ================================================================
    return X_norm, mu, sigma


# Execute the next cell to run the implemented `featureNormalize` function.

# In[18]:


# call featureNormalize on the loaded data
X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)


# After the `featureNormalize` function is tested, we now add the intercept term to `X_norm`:

# In[19]:


# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)


# <a id="section5"></a>
# ### 2.2 Gradient Descent
# 
# Previously, you implemented gradient descent on a univariate regression problem. The only difference now is that there is one more feature in the matrix $X$. The hypothesis function and the batch gradient descent update
# rule remain unchanged. 
# 
# You should complete the code for the functions `computeCostMulti` and `gradientDescentMulti` to implement the cost function and gradient descent for linear regression with multiple features. If your code in the previous part (single feature) already supports multiple features, you can use it here too.
# Make sure your code supports any number of features and is well-vectorized.
# You can use the `shape` property of `numpy` arrays to find out how many features are present in the dataset.
# 
# <div class="alert alert-block alert-warning">
# **Implementation Note:** In the multivariate case, the cost function can
# also be written in the following vectorized form:
# 
# $$ J(w) = \frac{1}{2m}(Xw - \vec{y})^T(Xw - \vec{y}) $$
# 
# where 
# 
# $$ X = \begin{pmatrix}
#           - (x^{(1)})^T - \\
#           - (x^{(2)})^T - \\
#           \vdots \\
#           - (x^{(m)})^T - \\ \\
#         \end{pmatrix} \qquad \mathbf{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \\\end{bmatrix}$$
# 
# the vectorized version is efficient when you are working with numerical computing tools like `numpy`. If you are an expert with matrix operations, you can prove to yourself that the two forms are equivalent.
# </div>
# 
# <a id="computeCostMulti"></a>

# In[20]:


def computeCostMulti(X, y, w):
    """
    Compute cost for linear regression with multiple features.
    Computes the cost of using w as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    w : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    
    Instructions
    ------------
    Compute the cost of a particular choice of w. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    
    J = sum((np.dot(X,w) - y)**2) / (2*m)
    
    # ==================================================================
    return J


# In[21]:


def gradientDescentMulti(X, y, w, alpha, num_iters):
    """
    Performs gradient descent to learn w.
    Updates w by taking num_iters gradient steps with learning rate alpha.
        
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    w : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    w : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector w.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    w = w.copy()
    
    J_history = []
    
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
         w = w - alpha/ m*(np.dot(X,w)-y)@X
        # w = w - alpha / m * np.dot((np.dot(X, w) - y),X)
        # =================================================================
        # save the cost J in every iteration
         J_history.append(computeCostMulti(X, y, w))
    
    return w, J_history


# #### Selecting learning rates
# 
# In this part of the exercise, you will get to try out different learning rates for the dataset and find a learning rate that converges quickly. You can change the learning rate by modifying the following code and changing the part of the code that sets the learning rate.
# 
# Use your implementation of `gradientDescentMulti` function and run gradient descent for about 50 iterations at the chosen learning rate. The function should also return the history of $J(w)$ values in a vector $J$.
# 
# After the last iteration, plot the J values against the number of the iterations.
# 
# If you picked a learning rate within a good range, your plot look similar as the following Figure. 
# 
# ![](Figures/learning_rate.png)
# 
# If your graph looks very different, especially if your value of $J(w)$ increases or even blows up, adjust your learning rate and try again. We recommend trying values of the learning rate $\alpha$ on a log-scale, at multiplicative steps of about 3 times the previous value (i.e., 0.3, 0.1, 0.03, 0.01 and so on). You may also want to adjust the number of iterations you are running if that will help you see the overall trend in the curve.
# 
# <div class="alert alert-block alert-warning">
# **Implementation Note:** If your learning rate is too large, $J(w)$ can diverge and ‘blow up’, resulting in values which are too large for computer calculations. In these situations, `numpy` will tend to return
# NaNs. NaN stands for ‘not a number’ and is often caused by undefined operations that involve −∞ and +∞.
# </div>
# 
# <div class="alert alert-block alert-warning">
# **MATPLOTLIB tip:** To compare how different learning learning rates affect convergence, it is helpful to plot $J$ for several learning rates on the same figure. This can be done by making `alpha` a python list, and looping across the values within this list, and calling the plot function in every iteration of the loop. It is also useful to have a legend to distinguish the different lines within the plot. Search online for `pyplot.legend` for help on showing legends in `matplotlib`.
# </div>
# 
# Notice the changes in the convergence curves as the learning rate changes. With a small learning rate, you should find that gradient descent takes a very long time to converge to the optimal value. Conversely, with a large learning rate, gradient descent might not converge or might even diverge!
# Using the best learning rate that you found, run the script
# to run gradient descent until convergence to find the final values of $w$. Next,
# use this value of $w$ to predict the price of a house with 1650 square feet and
# 3 bedrooms. You will use value later to check your implementation of the normal equations. Don’t forget to normalize your features when you make this prediction!

# In[22]:


"""
Instructions
------------
We have provided you with the following starter code that runs
gradient descent with a particular learning rate (alpha). 

Your task is to first make sure that your functions - `computeCost`
and `gradientDescent` already work with  this starter code and
support multiple features.

After that, try running gradient descent with different values of
alpha and see which one gives you the best result.

Finally, you should complete the code at the end to predict the price
of a 1650 sq-ft, 3 br house.

Hint
----
At prediction, make sure you do the same feature normalization.
"""
# Choose some alpha value - change this
alpha = 0.1
num_iters = 50

# init w and run gradient descent
w = np.zeros(3)
w, J_history = gradientDescentMulti(X, y, w, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')

# Display the gradient descent's result
print('w computed from gradient descent: {:s}'.format(str(w)))

# Estimate the price of a 1650 sq-ft, 3 br house
# ======================= YOUR CODE HERE ===========================
# Recall that the first column of X is all-ones. 
# Thus, it does not need to be normalized.

x_temp = [1650,3]
x_temp = (x_temp-mu)/sigma
x_temp = np.concatenate([[1], x_temp])
price =  np.dot(x_temp,w)  # You should change this

# ===================================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))


# <a id="section7"></a>
# 
# ## PART 3: Linear regression (multiple features, Normal Equation)
# 
# In the lecture, you learned that the closed-form solution to linear regression is Normal Equation
# 
# $$ w = \left( X^T X\right)^{-1} X^T\vec{y}$$
# 
# Using this formula does not require any feature scaling, and you will get an exact solution in one calculation: there is no “loop until convergence” like in gradient descent. 
# 
# First, we will reload the data to ensure that the variables have not been modified. Remember that while you do not need to scale your features, we still need to add a column of 1’s to the $X$ matrix to have an intercept term ($w_0$). The code in the next cell will add the column of 1’s to X for you.

# In[23]:


# Load data
data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)


# Complete the code for the function `normalEqn` below to use the formula above to calculate $w$. 
# 
# <a id="normalEqn"></a>

# In[24]:


def normalEqn(X, y):
    """
    Computes the closed-form solution to linear regression using the normal equations.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        The value at each data point. A vector of shape (m, ).
    
    Returns
    -------
    w : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).
    
    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.
    
    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    """
    w = np.zeros(X.shape[1])
    
    # ===================== YOUR CODE HERE ============================
    #w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    w = np.linalg.inv((X.T@X))@X.T@y
    # =================================================================
    return w


# In[25]:


# Calculate the parameters from the normal equation
w = normalEqn(X, y);

# Display normal equation's result
print('w computed from the normal equations: {:s}'.format(str(w)));

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
temp_x = [1,1650,3]
price = np.dot(temp_x,w) # You should change this

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))


# ## Submit the solution
# 
# When you completed the excercise, download (form File menu) this file as a jupyter Notebook file (.ipynb) and uplaod this file in the CANVAS 
# 
# *By writing down our names we declare that we have done the assignements ourselevs:*
# 
# * First Name  Last Name:
# * First Name  Last Name: 

# In[ ]:




