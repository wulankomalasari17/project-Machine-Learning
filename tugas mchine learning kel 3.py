# Import library yang dibutuhkan
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Database
FileDB = 'sinus.txt'
Database = pd.read_csv(FileDB, sep=",", header=0)
print("---------------------")
print(Database)

# x = data, y = target
x = Database[['Feature']]  # ciri1, ciri2, dst
y = Database['Target']

# Fit model regresi Nearest Neighbors
reg = KNeighborsRegressor(n_neighbors=3)  # Mengubah jumlah tetangga menjadi 3 (sesuaikan sesuai kebutuhan)
reg.fit(x, y)

# Menampilkan data prediksi
xx = np.arange(1, 21, 1)  # (data pertama, data terakhir, rentang)
n = len(xx)
print("xx(i)  K-NN")
for i in range(n):
    y_neighbor = reg.predict([[xx[i]]])
    print('{:.2f}'.format(xx[i]), y_neighbor[0])

# Plot data prediksi
y_neighbor2 = reg.predict(x)
plt.figure()
plt.plot(x, y_neighbor2, color='red')
plt.scatter(x, y, color='blue')
plt.title('Prediksi Data Menggunakan K-Nearest Neighbors')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['nearest neighbors', 'data'], loc=2)
plt.show()
