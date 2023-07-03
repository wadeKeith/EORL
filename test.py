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

x = np.linspace(0,50,100)
y = -(x-30)**2/(30)**2

plt.figure(0)
plt.plot(x,y)
plt.show()