from __future__  import division, print_function, unicode_literals
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
import matplotlib.pyplot as plt
# phong ngu
x1 = np.array([[3,4,3,3,3,4,4,4,3,4,3,4,4,3,5,4,3,3,5,3]]).T
#phong tam
x2 = np.array([[1,3,2,2.5,1,2.5,2.25,2.5,1.75,2,1.75,2.5,2.5,1.75,1.75,2.5,2.25,2.25,2.5,1]]).T
# dien tich
x3 = np.array([[143.0,295.0,171.0,232.0,109.0,262.0,422.0,225.0,126.0,275.0,179.0,190.0,198.0,196.0,203.0,323.0,291.0,117.0,282.0,106.0]]).T
# gia
y = np.array([[404.38904227,536.35713217,411.1227508,490.59197382,349.33626488,502.22549054,770.94835239,442.31511514,347.90974893,542.57701833,433.7273137,385.6431384,398.59673309,461.2537024,398.70317107,600.99664999,595.77570282,314.03501849,497.66701391,344.47866687]]).T
# hien thi gia - dien tich
plt.plot(x3, y, 'ro')
plt.axis([100.0, 400.0, 200, 800])
plt.xlabel("dien tich m2")
plt.ylabel("ti usd")
plt.show()
#
X = np.concatenate((x1, x2, x3), axis = 1)
one = np.ones((x1.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# w
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
# trong so hoi quy
print('w = ' , w)
# y = w1*x + w0
y_pred = np.dot(Xbar, w)
print('Dự đoán giá (y_pred):', y_pred)
print('Giá thực tế (y):', y)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'Sai số bình phương trung bình (MSE): {mse}')
print(f'R-squared: {r2}')
print(np.sum((y - y_pred)**2))