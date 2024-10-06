import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print('For make instance , model = slr(x, y, learning rate, literation, instance name)')



def prediction(a, b, x):
    return a * x + b

def update_ab(a, b, x, error, lr):
    delta_a = -(lr * (2 / len(error)) * np.dot(x.T, error))
    delta_b = -(lr * (2 / len(error)) * np.sum(error))
    return delta_a, delta_b

def gradient_descent(x, y, itr, lr):
    a = np.zeros((1, 1))
    b = np.zeros((1, 1))
    
    for i in range(itr):
        error = y - prediction(a, b, x)
        a_delta, b_delta = update_ab(a, b, x, error, lr)
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
    t_a = a[0, 0] / se_a
    t_b = b[0, 0] / se_b
    
    # 자유도
    df = n - 2
    
    # p-value 계산 (이중 측정이므로 두 배로 계산)
    p_value_a = 2 * (1 - stats.t.cdf(np.abs(t_a), df))
    p_value_b = 2 * (1 - stats.t.cdf(np.abs(t_b), df))
    
    return p_value_a, p_value_b

def calculate_r_squared(y, y_pred):
    # 잔차 제곱합
    ss_res = np.sum((y - y_pred) ** 2)
    # 총 제곱합
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    # R^2 계산
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

class slr:
    
    def __init__(self, x, y, lr, iteration, instance_name):
        self.x = x
        self.y = y
        self.lr = lr
        self.itr = iteration
        self.instance_name = instance_name
        self.a, self.b = gradient_descent(self.x, self.y, self.itr, self.lr)
  
    def setdata(self, x, y, lr, iteration, instance_name):
        self.x = x
        self.y = y
        self.lr = lr
        self.itr = iteration
        self.instance_name = instance_name
        self.a, self.b = gradient_descent(self.x, self.y, self.itr, self.lr)
			
    def eq(self):
        print(f"a: {self.a}, b: {self.b}, Equation : {self.a}*x+{self.b}")
        return self.a, self.b
    
    def p_value(self):
        p_value_a, p_value_b = calculate_p_value(self.x, self.y, self.a, self.b)
        return p_value_a, p_value_b
        
    def r_squared(self):
        y_pred = prediction(self.a, self.b, self.x)
        r_squared = calculate_r_squared(self.y, y_pred)
        return r_squared
    
    def plotting_graph(self):
        plt.figure()
        y_pred = prediction(self.a, self.b, self.x)
        r_squared = calculate_r_squared(self.y, y_pred)
        p_value_a, p_value_b = calculate_p_value(self.x, self.y, self.a, self.b)
    
        plt.scatter(self.x, self.y, label='Actual data', color='blue')
        plt.plot(self.x, y_pred, label='Predicted line', color='red')
    
        equation_text = f'y = {self.a[0, 0]:.2f} * x + {self.b[0, 0]:.2f}\nR² = {r_squared:.3f}'
        plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
        p_value_text = f'p-value (a): {p_value_a:.3g}\np-value (b): {p_value_b:.3g}'
        plt.text(0.05, 0.75, p_value_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression Fit')
        plt.legend()
        file_name = f"{self.instance_name}_plot.png"
        plt.savefig(file_name)
        plt.show()