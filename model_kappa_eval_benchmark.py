from numpy import array

from matplotlib.pyplot import boxplot
from matplotlib.pyplot import figure
from matplotlib.pyplot import show

from pickle import load

from einstein.plot.plots import BoxPlot



plt = BoxPlot(name="model_kappa_eval_benchatk", figsize=(7,8))
plt.title = "cartpole problem benchmark"
plt.x_label = "real worlds samples"
plt.y_label = "reward"
plt.plot("/home/deepthree/Desktop/deepcontrol/records_kappa_eval.p", x_labels=range(2, 3003, 200))