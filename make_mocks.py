import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import normalize,MinMaxScaler
import matplotlib

# make baseline of counts
freq = 1.
num = int(24*3600*freq)
days = 30
for i in range(10000):
    # where should the earth quake happen on the last day
    Eq = np.random.uniform(low=num/2,high=num)+(days-1)*num
    #Magnitude should increase toward earthquake
    for d in range(days):
        s = np.random.standard_normal(num)
        t = np.arange(num)+d*num
        alpha = 3./np.log(num*days)
        linear = alpha*np.log(t+1)
        strength = 2./np.log(num*days)*np.log(t+1)+1
        norm = np.random.standard_normal(num)*strength
        norm_pois = np.random.poisson(lam=linear)
        norm[t>Eq] = 0
        norm[norm_pois <=8] = 0
        data = s+norm
        np.save('/home/ashking/quake_finder/data/mocks/EQ'+str(i)+'_'+str(days-d-1)+'daysuntilEQ.npy',data)

