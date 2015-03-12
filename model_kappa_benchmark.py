from numpy import array

from matplotlib.pyplot import boxplot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

from pickle import load

from einstein.plot.plots import BoxPlot



plt = BoxPlot(name="model_eta_benchatk")
plt.title = "cartpole problem benchmark"
plt.x_label = "time steps"
plt.y_label = "reward"
plt.plot("/home/deepthree/Desktop/deepcontrol/records.p")