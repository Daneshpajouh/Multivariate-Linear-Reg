%reset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from urllib.request import urlopen
y_train = np.reshape((np.array(np.loadtxt((urlopen('https://raw.githubusercontent.com/Daneshpajouh/Multivariate-Linear-Reg/master/y_train.csv')), delimiter = ','))), (400,1))
x_train = np.c_[np.ones(len(y_train)), (np.array(np.loadtxt((urlopen('https://raw.githubusercontent.com/Daneshpajouh/Multivariate-Linear-Reg/master/x_train.csv')), delimiter = ',')))]
y_test = np.reshape((np.array(np.loadtxt((urlopen('https://raw.githubusercontent.com/Daneshpajouh/Multivariate-Linear-Reg/master/y_test.csv')), delimiter = ','))), (100,1))
x_test = np.c_[np.ones(len(y_test)), (np.array(np.loadtxt((urlopen('https://raw.githubusercontent.com/Daneshpajouh/Multivariate-Linear-Reg/master/x_test.csv')), delimiter = ',')))]

# Initializing required values
theta = np.zeros((3,1))
alpha = 0.2
m = len(y_train)
num_itr = 50
epsilon = 1e-3
cost = []

for i in range(num_itr):   # Main loop
    h_x = (x_train @ theta)
    error = float(((h_x - y_train).transpose() @ (h_x - y_train))/(2 * m))      # Calculating error in each iteration
    cost.append(error)
    theta = theta - ((alpha / m) * ((x_train.transpose())@(h_x - y_train)))     # Training Theta via "Gradient Descent function"
    print(error)
    if i < 5:
        pass
    elif abs(np.mean(cost[-5:]) - error) <= epsilon:     # Break the loop if error is too small or not changing (Convergence)
        break
        
hypo = x_test @ theta       #  Testing the hypothesis
print("\n\nTheta:\n\n", theta, '\n\nTest Values:\n\n', hypo)

# Cost function figure
fig, ax = plt.subplots()
ax.scatter(range(len(np.array(cost).transpose())), np.array(cost).transpose(), marker="x", c="red")
plt.title("Cost", fontsize=16)
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Error", fontsize=14)
plt.axis([0, int(len(np.array(cost).transpose())), 0.042 , float(cost[0])])
ax.plot(range(len(np.array(cost).transpose())), np.array(cost).transpose(), linewidth=2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x_test[:,1], x_test[:,2], c=x_test[:,2], cmap='Greens');
plt.title("Dataset", fontsize=16)
ax.set_xlabel('X1', fontsize=14)
ax.set_ylabel('X2', fontsize=14)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x_test[:,1], x_test[:,2], y_test, c=x_test[:,2], cmap='Greens');
plt.title("Dataset", fontsize=16)
ax.set_xlabel('X1', fontsize=14)
ax.set_ylabel('X2', fontsize=14)
ax.set_zlabel('Y', fontsize=14)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x_test[:,1], x_test[:,2], hypo, c=x_test[:,2], cmap='Greens');
plt.title("Dataset", fontsize=16)
ax.set_xlabel('X1', fontsize=14)
ax.set_ylabel('X2', fontsize=14)
ax.set_zlabel('Y', fontsize=14)
ax.plot_wireframe(np.arange(0.0,1.0,0.01), np.arange(0.0,1.0,0.01), y_test, rstride=10, cstride=10)
plt.show()
