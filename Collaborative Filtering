Data available at https://drive.google.com/drive/folders/0B0RLknmL54khU2UwX3dnX1E1WHc?usp=sharing

Movie1 Movie2 Movie 3  Genre
[[  3.,   4.,   4.,   3.],
 [  2.,   3.,   3.,   3.],
 [  4.,   5.,   5.,   3.],
 [  1.,   2.,   2.,   3.],
 [  4.,   5.,  nan,   3.],
 [  4.,   5.,   5.,   3.],
 [  2.,   3.,   3.,   4.],
 [  0.,   1.,   1.,   4.],
 [  0.,   1.,   1.,   4.],
 [  0.,   1.,   1.,   4.],
 [  4.,   5.,   5.,   4.],
 [  1.,   2.,   2.,   4.],
 [  4.,   3.,   3.,   5.],
 [  4.,   3.,   3.,   5.],
 [  2.,   1.,   1.,   5.],
 [  3.,   2.,   2.,   5.],
 [  3.,   2.,   2.,   5.],
 [  2.,   1.,   1.,   5.],
 [  5.,   4.,   4.,   5.]]

import pandas as pd
import numpy as np

aa=pd.read_csv('Collab.Filtering.Data.csv',sep=',',header=0)

missing_value=5

d=[]
for i in range(0,aa.shape[1]):
    d.append(pd.isnull(aa[[i]]).any(1).nonzero()[0])

column=np.where(np.array(d)>0)[0][0]
row=d[column][0]
subset=aa[:row+1]

minim=min(np.where(aa[[3]]==3)[0])
maxim=max(np.where(aa[[3]]==3)[0])+1
b=aa.fillna(0)
col=np.array(b[[2]][minim:maxim]).T
row_=np.array(b)
selected_rows=row_[minim:maxim]
row_calculation=row_[row]

ab=[]
for i in range(0,6):
    x=(col[0][i]-np.mean(selected_rows[i][0:3]))
    y=(col[0][i]-np.mean(col[0]))
    ab.append(x*y/np.sqrt(x**2*y**2))
ab

predicted_value=np.mean(row_calculation)+np.sum(ab)

print('Accuracy:',missing_value/predicted_value)
