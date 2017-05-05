# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:41:49 2017

@author: Doug_Hatfield
"""

def clean_data( df, remove_cols_containing_missing ):
    # Take out text eg. sample labels4
    print("Removing columns containing text...")
    print("")
    df2 = df.select_dtypes(include=['number']).copy()

    # Identify columns containing no data
    emptycolumns = df.columns[df.isnull().all()]
    df2.drop(emptycolumns, axis = 1, inplace = True)
    print("Empty columns dropped:")
    print(emptycolumns)
        
    if (remove_cols_containing_missing):
        # Remove columns containing missing data (NaNs)
        badcolumns = df2.columns[df2.isnull().any()]
        df2.drop(badcolumns, axis = 1, inplace = True)
        print("")
        print("Columns containing NaNs in input:")
        print(badcolumns)

    zero_stdev_columns = df2.columns[df2.std(axis = 0, skipna = False) < 1e-10]
    print("")
    print("Zero stdev columns:")
    print(zero_stdev_columns)

    df2.drop(zero_stdev_columns, axis=1, inplace = True)
    
    # Drop rows or columns that contain no data
    df2.dropna(how = 'all', axis = 0, inplace = True)
    df2.dropna(how = 'all', axis = 1, inplace = True)
    
    return df2
    


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#%matplotlib inline

meuse = pd.read_csv(filepath_or_buffer='C:\\My Documents\\Work\\Data science\\Geostat Meuse\\meuse.csv',
           sep=',')
meuse = clean_data(meuse, False)
meuse_polygon = np.array(pd.read_csv('C:\\My Documents\\Work\\Data science\\Geostat Meuse\\meuse_polygon.csv', sep=','))

#fig = plt.figure(figsize=(12,8))
#ax1 = fig.add_subplot(121, aspect=1)
#ax1.set_title('Lead')
#ax1.plot(meuse_polygon[:, 0], meuse_polygon[:, 1], 'k-')
#ax1.scatter(x=meuse.x, y=meuse.y, s=meuse.lead, alpha=0.5, color='grey')
#
#ax2 = fig.add_subplot(122, aspect=1)
#ax2.set_title('Copper')
#ax2.plot(meuse_polygon[:, 0], meuse_polygon[:, 1], 'k-')
#ax2.scatter(x=meuse.x, y=meuse.y, s=meuse.copper, alpha=0.5, color='orange')
#
#fig = plt.figure(figsize=(12,8))
#plt.plot(np.log10(meuse['lead']), np.log10(meuse['copper']), '.')
#plt.xlabel('Log10(Lead)')
#plt.ylabel('log10(Copper)')

import itertools
def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

grid = expand_grid({'X1': np.linspace(178605, 181390, 10),
                  'X2': np.linspace(329714, 333611, 10)})
grid = np.array(grid)

nans = np.full((len(grid), 11),np.NaN)
gridNaN = np.concatenate((grid, nans), axis = 1)

#from sklearn.cross_validation import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(meuse[['x', 'y']], np.log10(meuse[['lead', 'copper']]), test_size=0.3, random_state=747475)

#Y, test = np.array(train_test_split(meuse.values, test_size=0.3, random_state=747475))
Y = meuse.values


## Coords
#X1 = np.array(X_train)[:,0][:,None]
#X2 = np.array(X_train)[:,1][:,None]
## log10(Pb) and log10(Cu)
#Y1 = np.array(Y_train)[:,0][:,None]
#Y2 = np.array(Y_train)[:,1][:,None]

#Y = np.concatenate((X2, X1, Y1, Y2), axis = 1)

Y = np.concatenate((Y, gridNaN), axis = 0)
Y_in = np.array(Y)

# Normalise
Y_mean = np.nanmean(Y,0)
Y -= Y_mean

Y_std = np.nanstd(Y,0)
Y /= Y_std

import GPy

# BGPLVM config
input_dim = 4
clusters = input_dim + 1
NUM_ITERS = 2000

# BGPLVM persisting
LOAD_GPM = True
    
m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(Y, input_dim, 
                                                           num_inducing=clusters,
                                                           missing_data=True, 
                                                           init = "PCA",
                                                           batchsize = 100,
                                                           stochastic = False,
                                                           initialize = True)
print(m)

m.optimize('bfgs', max_iters = NUM_ITERS, ipython_notebook = True, messages = True) #first runs EP and then optimizes the kernel parameters
print(m)  

# The embedded data is in the latent space, so not useful for interpretation
data_embedded = m.X.mean.values

# Predict the values in original data space
fitted_data = m.predict(data_embedded)

# Reverse the normalisation
Y_out = fitted_data[0] * Y_std + Y_mean
Y_out[:,2] = np.power(10, Y_out[:,2])
Y_out[:,3] = np.power(10, Y_out[:,3])

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(121, aspect=1)
ax1.set_title('Lead')
ax1.plot(meuse_polygon[:, 0], meuse_polygon[:, 1], 'k-')
#ax1.scatter(x=meuse.x, y=meuse.y, s=meuse.lead, alpha=0.5, color='grey')
ax1.scatter(x=Y_in[:,1], y=Y_in[:,0], s=np.power(10, Y_in[:,2]), alpha=0.5, color='grey')

ax2 = fig.add_subplot(122, aspect=1)
ax2.set_title('Lead fit')
ax2.plot(meuse_polygon[:, 0], meuse_polygon[:, 1], 'k-')
ax2.scatter(x=Y_in[:,1], y=Y_in[:,0], s=Y_out[:,2], alpha=0.5, color='blue')
#ax2.scatter(x=Y_out[:,1], y=Y_out[:,0], s=Y_out[:,2], alpha=0.5, color='blue')

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(121, aspect=1)
ax1.set_title('Copper')
ax1.plot(meuse_polygon[:, 0], meuse_polygon[:, 1], 'k-')
#ax1.scatter(x=meuse.x, y=meuse.y, s=meuse.lead, alpha=0.5, color='grey')
ax1.scatter(x=Y_in[:,1], y=Y_in[:,0], s=np.power(10, Y_in[:,3]), alpha=0.5, color='grey')

ax2 = fig.add_subplot(122, aspect=1)
ax2.set_title('Copper fit')
ax2.plot(meuse_polygon[:, 0], meuse_polygon[:, 1], 'k-')
ax2.scatter(x=Y_in[:,1], y=Y_in[:,0], s=Y_out[:,3], alpha=0.5, color='orange')
#ax2.scatter(x=Y_out[:,1], y=Y_out[:,0], s=Y_out[:,2], alpha=0.5, color='blue')

