import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

learning_rate = 1e-5
iteration = 10000

x = np.array([[8.70153760], [3.90825773], [1.89362433], [3.28730045], 
[7.39333004], [2.98984649], [2.25757240], [9.84450732], [9.94589513], [5.48321616]])
y = np.array([[5.64413093], [3.75876583], [3.87233310], [4.40990425], 
[6.43845020], [4.02827829], [2.26105955], [7.15768995], [6.29097441], [5.19692852]])

def prediction(a, b, x):
    return a * x + b

def update_ab(a, b, x, error, lr):
    delta_a = -(lr * (2 / len(error)) * np.dot(x.T, error))
    delta_b = -(lr * (2 / len(error)) * np.sum(error))
    return delta_a, delta_b

def gradient_descent(x, y, iteration):
    a = np.zeros((1,1))
    b = np.zeros((1,1))
    
    for i in range(iteration):
        error = y - prediction(a, b, x)
        a_delta, b_delta = update_ab(a, b, x, error, lr=learning_rate)
        a -= a_delta
        b -= b_delta
    
    return a, b

def calculate_p_value(x, y, a, b):
    n = len(y)
    
    # 예측 값 계산
    y_pred = prediction(a, b, x)
    
    # 잔차 계산
    residuals = y - y_pred
    
    # 잔차의 분산
    residual_sum_of_squares = np.sum(residuals**2)
    residual_variance = residual_sum_of_squares / (n - 2)
    
    # x의 분산
    x_mean = np.mean(x)
    s_xx = np.sum((x - x_mean) ** 2)
    
    # a와 b의 표준 오차 계산
    se_a = np.sqrt(residual_variance / s_xx)
    se_b = np.sqrt(residual_variance * (1/n + (x_mean ** 2) / s_xx))
    
    # t-값 계산
    t_a = a / se_a
    t_b = b / se_b
    
    # 자유도
    df = n - 2
    
    # p-value 계산 (이중 측정이므로 두 배로 계산)
    p_value_a = 2 * (1 - stats.t.cdf(np.abs(t_a), df))
    p_value_b = 2 * (1 - stats.t.cdf(np.abs(t_b), df))
    
    return p_value_a, p_value_b

def plotting_graph(x, y, a, b):
    y_pred = a[0, 0] * x + b
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.savefig("test.png")

def main():
    a, b = gradient_descent(x, y, iteration)
    p_value_a, p_value_b = calculate_p_value(x, y, a, b)
    
    print(f"a: {a}, b: {b}")
    print(f"p-value for a: {p_value_a}, p-value for b: {p_value_b}")
    
    plotting_graph(x, y, a, b)

main()
