# find all the days with at least X number of days before them to use in
# training and testing

import numpy as np
import os

lookback = 5
PATH='/home/ashking/quake_finder/data_qf/'

def find_lookback(PATH,day,lookback):
    """ determine if all the days are present in the file"""
    path_n = np.copy(PATH)
    str(path_n).replace('d_mag','n_mag')
    path_e = np.copy(PATH)
    str(path_e).replace('d_mag','e_mag')
    times = Time(day[:4]+'-'+day[4:6]+'-'+day[6:8])
    t = Time(times)
    for l in range(lookback):
        delta_t = TimeDelta(l,format='jd')
        previous_day = (t-delta_t).value
        previous_day = previous_day[:4]+previous_day[5:7]+previous_day[8:10]
        if not os.path.isfile(PATH+'/'+previous_day+'.npy'):
            return False
        elif not os.path.isfile(str(path_n)+'/'+previous_day+'.npy'):
            return False
        elif not os.path.isfile(str(path_e)+'/'+previous_day+'.npy'):
            return False
    return True


filename = ['','_validate']
for i in range(2):
    f = open(PATH+'list_of_'+filename[i]+'data_lookback'+str(lookback)+'.txt','w')
    with open(PATH+'list_of'+filename[i]+'_data.txt','r') as fp:
        line = fp.readline()
        while line:
            path = line[:-14]
            day = line[-13:-5]
            path_n = np.copy(path)
            str(path_n).replace('d_mag','n_mag')
            path_e = np.copy(path)
            str(path_e).replace('d_mag','e_mag')
            if find_lookback(PATH+path,day,lookback):
                if find_lookback(PATH+str(path_n),day,lookback):
                    if find_lookback(PATH+str(path_e),day,lookback):
                        f.write(line)
            line = fp.readline()
    f.close()
