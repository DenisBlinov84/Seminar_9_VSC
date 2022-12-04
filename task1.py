# Постройте графики для приведённых наборов данных.
# Найдите коэффициенты для линии регрессии и коэффициенты детерминации.
# Что вы замечаете?
# Нанесите на график модель линейной регрессии.
# X1 = np.array([30,30,40,40])
# Y1 = np.array([37,47,50,60])

from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([30, 30, 40, 40])
y1 = np.array([37, 47, 50, 60])


model = LinearRegression()
x1 = x1.reshape(-1, 1)
model.fit(x1, y1)
r_sq = model.score(x1, y1)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
# coefficient of determination: 0.6282527881040892
# intercept: 3.0
# slope: [1.3]

plt.scatter(x1, y1)
plt.plot(x1, 3+1.3*x1)
plt.show()

y_hat = 3 + 1.3 * x1
y_hat = y_hat.reshape(1, -1)
res = y1 - y_hat
print(stats.shapiro(res))
# ShapiroResult(statistic=0.728634238243103, pvalue=0.02385682798922062)
plt.scatter(y_hat, res)
plt.show()
