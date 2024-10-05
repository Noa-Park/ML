import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

learning_rate = 1e-4
iteration = 10000

x = np.array([[8.70153760], [3.90825773], [1.89362433], [3.28730045], 
[7.39333004], [2.98984649], [2.25757240], [9.84450732], [9.94589513], [5.48321616]])
y = np.array([[5.64413093], [3.75876583], [3.87233310], [4.40990425], 
[6.43845020], [4.02827829], [2.26105955], [7.15768995], [6.29097441], [5.19692852]])

def prediction(a,b,x):
    eq = a*x + b
    return eq

def update_ab(a,b,x,error,lr):
    delta_a = -(lr*(2/len(error))*(np.dot(x.T, error)))
    delta_b = -(lr*(2/len(error))*np.sum(error))
    
    return delta_a, delta_b

def gradient_descent(x, y, iteration):

    a = np.zeros((1,1))
    b = np.zeros((1,1))
    
    for i in range(iteration):
        error = y - prediction(a,b,x)
        a_delta, b_delta = update_ab(a,b,x,error, lr = learning_rate)
        a -= a_delta
        b -= b_delta
    
    return a, b

def plotting_graph(x,y,a,b):
    y_pred=a[0,0]*x+b
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.savefig("test.png")

def main():
    a, b = gradient_descent(x, y, iteration)
    t, pvalue = stats.ttest_ind(y, x)
    print("a:",a, "b:",b, "p-value:", pvalue)
    plotting_graph(x,y,a,b)
    
main()