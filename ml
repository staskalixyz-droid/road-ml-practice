import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


csv_data = """Country,GDP per capita,Life satisfaction
Hungary,12240,4.9
Korea,27195,5.8
France,37675,6.5
Australia,50962,7.3
United States,55805,7.2
"""

with open("life_data.csv", "w", encoding="utf-8") as file:
    file.write(csv_data)


country_stats = pd.read_csv("life_data.csv")


X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]


country_stats.plot(
    kind="scatter",
    x="GDP per capita",
    y="Life satisfaction"
)

plt.show()


model = LinearRegression()
model.fit(X, y)


plt.scatter(X, y)

X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)

plt.plot(X_line, y_line)

plt.xlabel("GDP per capita")
plt.ylabel("Life satisfaction")
plt.show()


X_new = [[22587]]
prediction = model.predict(X_new)

print("Прогноз для GDP per capita = 22587:")
print(prediction)