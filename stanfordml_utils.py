import numpy as np

def sumOfSquares(predict, result):
    return sum((1.0/(2*predict.shape[0]))*((y-t)**2) for y, t in zip(predict, result))
