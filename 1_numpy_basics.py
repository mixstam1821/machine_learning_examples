import numpy as np

vector = np.array([1, 2, 3, 4])
print("Vector: {}".format(vector))
# Every array will have a shape. That is, its dimensions
print("Shape: {}".format(vector.shape))
# Print number of dimensions
print("Dim: {}".format(vector.ndim))
print("Data type: {}".format(vector.dtype))



v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
v.shape = (2, 3, 2)
print(v)




v = np.zeros((2, 3, 2))
print(v)




a = np.arange(15)
print(a)



print("Zeros")
a = np.zeros((3, 3))
print("A: {}".format(a))
b = np.zeros_like(a)
print("B: {}".format(b))
print("\nOnes")
a = np.ones((3, 3))
print("A: {}".format(a))
b = np.ones_like(a)
print("B: {}".format(b))
print("\nEmpty")
a = np.empty((3, 3))
print("A: {}".format(a))
b = np.empty_like(a)
print("B: {}".format(b))





a = np.array([1, 2, 3, 4.5, 6.7])
print("A: {}, dtype: {}".format(a, a.dtype))
b = a.astype(np.int)
print("B: {}, dtype: {}".format(b, b.dtype))






a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[4, 5, 6], [1, 2, 3]])

c = a + b
print(c)

c = a * b
print(c)

c = a - b
print(c)



a = np.arange(20)
print(a)
a[10:15] = 5
print(a)




a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
print(a)
a[[1, 3, 2]]





a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
print(a.T)






xs = np.linspace(1, 10, 100)
ys = np.linspace(1, 10, 100)
xx, yy = np.meshgrid(xs, ys)






a = [0, -1, 2, 3, -4, -5]
b = [9, 3, 4, 11, 2, 3]
c = [True, False, True, True, False, True]
np.where(c, a, b)







a = np.random.rand(3, 3)
print(a)
print(np.mean(a)) # both are fine
print(a.mean())

print(np.std(a))







