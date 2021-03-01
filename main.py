import numpy as np
from pylab import rand
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def perceptronBinary(X, y, n):
    w = np.zeros(np.shape(X[0]))
    b = 0
    for i in range(1, n):
        for j in range(len(X)):
            if y[j] != np.sign(np.dot(w, X[j]) + b):
                w += np.dot(y[j], X[j])
                b += y[j]
    return w, b


#def perceptronMulti(X, y, n):


def score(X, y, w):
    error = 0
    for i in range(len(X)):
        if y[i] != np.sign(np.dot(w, X[i])):
            error += 1
    return 1 - error/len(X)


def genererDonnees(n):
    x1b = (rand(n) * 2 - 1) / 2 - 0.5
    x2b = (rand(n) * 2 - 1) / 2 + 0.5
    x1r = (rand(n) * 2 - 1) / 2 + 0.5
    x2r = (rand(n) * 2 - 1) / 2 - 0.5
    donnees = []
    for i in range(len(x1b)):
        donnees.append(((x1b[i], x2b[i]), False))
        donnees.append(((x1r[i], x2r[i]), True))
    return donnees


f = open("data.biais", "r")
line = f.readlines()[4:]
lines = [line[i][4:] for i in range(len(line))]
new_line = [lines[i].split('\n')[0][:-1] for i in range(len(lines))]

print(new_line[0][0])
"""
data = genererDonnees(1000)
X = [data[i][0] for i in range(len(data))]
y = [data[i][1] for i in range(len(data))]
y = [1 if x == True else -1 for x in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
w, b = perceptronBinary(X_train, y_train, 100)
scorea = score(X_train, y_train, w)
scoret = score(X_test, y_test, w)


print(scorea)
print(scoret)
print(b)

Xtrue = []
Xfalse = []

for i in range(len(X_test)):
    if y_test[i] == 1:
        Xtrue.append(X_test[i])
    else:
        Xfalse.append(X_test[i])

Xt = [Xtrue[i][0] for i in range(len(Xtrue))]
yt = [Xtrue[i][1] for i in range(len(Xtrue))]
Xf = [Xfalse[i][0] for i in range(len(Xfalse))]
yf = [Xfalse[i][1] for i in range(len(Xfalse))]

x = np.arange(-1, 1, 0.1)
y_ = w[0]*x/w[1]

plt.figure()
plt.scatter(Xt, yt, c='red')
plt.scatter(Xf, yf, c='blue')
plt.plot(x, -y_)
plt.show()
"""