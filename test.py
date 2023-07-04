import math
import numpy as np
import matplotlib.pyplot as plt
'''a = math.exp(-1)
x = np.linspace(0,50,100)
y = np.exp(-abs(x-30)/10)
plt.figure(0)
plt.plot(x,y)
plt.show()
print(a)'''

# x = np.linspace(0,50,100)
# y = 1-(x-30)**2/(30)**2
x = np.linspace(0,3.75*3,100)
y = -(x-3.75*3/2)**2/(3.75*3/2)**2/6

plt.figure(0)
plt.plot(x,y)
plt.show()