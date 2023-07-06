
for i in range(10):
    state_mean = 123
    state_std = 456
    ms = open('ms.txt', 'a+')
    ms.write('0'+' '+str(state_mean) +' '+str(state_std)+ '\n')
    ms.close()