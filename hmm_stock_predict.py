# learn the code from 
# https://hmmlearn.readthedocs.io/en/stable/auto_examples/plot_hmm_stock_analysis.html#sphx-glr-auto-examples-plot-hmm-stock-analysis-py

from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
import csv
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from hmmlearn.hmm import GaussianHMM

###############################################################################
# Get data from local cvs file
print("Extracting data...")

# data range 2018/6/11 - 2008/5/8
# testing data from 2018/1/11 - 2018/6/11; OR take data from 100 business days 
# training data from 2015/1/5 - 2018/1/10; 

counter = 0

# data structure [['2008-05-20', 'open', 'high', 'low', 'close', 'volume'], ['2008-05-08', 'open', 'high', 'low', 'close', 'volume'], ...]

# init the data and the first element is used to keep the same dimension
raw_data = np.array(['2000-01-01', '0', '0', '0', '0', '0'])

# read data from the csv file
with open('002236大华股份.csv', encoding="ISO-8859-1") as f:
	dahua = csv.reader(f)
	headers = next(dahua)
	for row in dahua:
		# print(row)
		curr_date = row[0]
		curr_close = row[3]
		curr_high = row[4]
		curr_low = row[5]
		curr_open = row[6]
		curr_volume = row[11]
		# print(curr_date, curr_close, curr_high, curr_low, curr_open)
		# 停牌的问题：take out the elements whose open is 0
		if curr_open == '0.00' or curr_open == '0.0' or curr_open == '0':
			pass
		else:
			temp_array = np.array([curr_date, curr_open, curr_high, curr_low, curr_close, curr_volume])	
			raw_data = np.vstack((raw_data, temp_array))
			# print(temp_array)
		

# get rid of the first meaningless element
raw_data = raw_data[1:]
# print(raw_data)

# 停牌的问题如何解决？？？
testing_data = raw_data[:200]
training_data = raw_data[201:]
# print(testing_data)
# print(training_data)

# 模仿链接里的代码 计算dates，close_v，volume
dates = np.array([array[0] for array in training_data])
# print(dates)
close_v = np.array([array[4] for array in training_data])
#convert str to float
close_v = close_v.astype(np.float)
# print(close_v)
volumes = np.array([array[5] for array in training_data])[1:]
#convert str to int
volumes = volumes.astype(np.int)
# print(volumes)

# Take diff of close value. Note that this makes
# ``len(diff) = len(close_t) - 1``, therefore, other quantities also
# need to be shifted by 1.
diff = np.diff(close_v)
# print(diff)
dates = dates[1:]
close_v = close_v[1:]

# Pack diff and volume for training.
X = np.column_stack([diff, volumes])
# print(X)
###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ...", end="")

# Make an HMM instance and execute fit
model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print("done")

###############################################################################
# print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()

