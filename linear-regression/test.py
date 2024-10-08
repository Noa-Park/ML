import numpy as np
import Apply_linear_regression as ml

x1 = np.array([[8.70153760], [3.90825773], [1.89362433], [3.28730045], 
              [7.39333004], [2.98984649], [2.25757240], [9.84450732], 
              [9.94589513], [5.48321616]])
y1 = np.array([[5.64413093], [3.75876583], [3.87233310], [4.40990425], 
              [6.43845020], [4.02827829], [2.26105955], [7.15768995], 
              [6.29097441], [5.19692852]])

model1 = ml.slr(x1, y1, lr=1e-4, iteration=10000, instance_name='model1')

a1, b1 = model1.eq() 
print(a1,b1)
p_val_a1, p_val_b1 = model1.p_value()
print(f"p-value a: {p_val_a1}, p-value b: {p_val_b1}")

r2_1= model1.r_squared()
print(f"R-squared: {r2_1}")

model1.plotting_graph()