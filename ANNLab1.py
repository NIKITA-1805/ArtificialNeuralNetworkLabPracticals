import numpy as nbp
import matplotlib.pyplot as plt

def sigmoid():
    x = nbp.linspace(-10, 10, 100)
    y = 1/(1+nbp.exp(-x))
    plt.xlabel('input')
    plt.grid(True)
    plt.plot(x, y)

def tanh():
    x = nbp.linspace(-10, 10, 100)
    y = nbp.tanh(x)
    plt.xlabel('input')
    plt.grid(True)
    plt.plot(x, y)

def binary():
    x = nbp.linspace(-10, 10, 100)
    y = nbp.where(x >= 0, 1, 0)
    plt.xlabel('input')
    plt.grid(True)
    plt.plot(x, y)

def relu():
    x = nbp.linspace(-10, 10, 100)
    y = nbp.maximum(0, x)
    plt.xlabel('input')
    plt.grid(True)
    plt.plot(x, y)

sigmoid()
plt.show()

tanh()
plt.show()

relu()
plt.show()

binary()
plt.show()