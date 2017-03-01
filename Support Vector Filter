from PIL import Image
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

im = Image.open("Van_GoghSmall_Blur.jpg").convert('LA')
plt.imshow(im)
trainX=np.array(im.getdata()).T[0].reshape(16500,1)

im2 = Image.open("Milla_JojoSmall.jpg").convert('LA')
trainY=np.array(im2.getdata()).T[0].reshape(16500,)

im3 = Image.open("Van_GoghSmall.jpg").convert('LA')
trainX2=np.array(im3.getdata()).T[0].reshape(16500,1)

im4 = Image.open("Milla_JojoSmall.jpg").convert('LA')

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=.002,max_iter= 1000000)
model = svr_rbf.fit(trainX,trainY)

trainPredict = model.predict(trainX2)

f, axarr = plt.subplots(2,2,figsize=(10,10))
axarr[0,0].axis('off')
axarr[0,0].imshow(im)
axarr[0,1].axis('off')
axarr[0,1].imshow(im2)
axarr[1,0].axis('off')
axarr[1,0].imshow(im3)
axarr[1,1].axis('off')
axarr[1,1].imshow(trainPredict.reshape(110,150), cmap=plt.get_cmap('gray'))

