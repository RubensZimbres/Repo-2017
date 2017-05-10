import cv2
img = cv2.imread('/Volumes/16 DOS/Python/StreetCars.png')
gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
image = gaussian_filter(gray, 1)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')
plt.imshow(dilated)

from skimage.filters import sobel
elevation_map = sobel(image)
plt.imshow(elevation_map)

markers = np.zeros_like(image)
markers[image < 100] = 1
markers[image > 150] = 2


### TREES
from skimage.morphology import watershed
segmentation = watershed(elevation_map, markers)
plt.imshow(segmentation)

####### FOURIER
'''http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html'''
f = np.fft.fft2(im2)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(im2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

###
rows, cols = im.shape
crow,ccol = int(rows/2) , int(cols/2)
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(121),plt.imshow(im, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(121),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.show()


# Calculate gradient 
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

plt.imshow(gy)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

plt.imshow(angle)

##

from skimage.filter import threshold_otsu, threshold_adaptive

global_thresh = threshold_otsu(gray0.astype(np.float32))
binary_global = gray0.astype(np.float32) > global_thresh
plt.imshow(binary_global)

block_size = 45
binary_adaptive = threshold_adaptive(gray0, block_size, offset=5)
plt.imshow(binary_adaptive)

from skimage.filters import gabor
from skimage import data, io
from matplotlib import pyplot as plt 

#### ESSE E FODA !!!!!
filt_real, filt_imag = gabor(gray0, frequency=2.1)
plt.figure()            
io.imshow(filt_real)    
io.show()   

plt.figure()            
io.imshow(filt_imag)    
io.show()   

##
from skimage.filters import gaussian, gaussian_filter, laplace,prewitt
from skimage.filters import prewitt_v,prewitt_h,scharr, wiener

gauss=gaussian(gray0, sigma=5, multichannel=True)
plt.imshow(gauss)

gauss2=gaussian_filter(gray0, sigma=5, multichannel=True)
plt.imshow(gauss2)

lap=laplace(gray0,ksize=100)
plt.imshow(lap)

pre=prewitt(gray0, mask=None)
plt.imshow(pre)

pre_v=prewitt_v(gray0, mask=None)
plt.imshow(pre_v)

from skimage import filters
edges2 = filters.roberts(gray0)
plt.imshow(edges2)

plt.imshow(scharr(gray0))
plt.imshow(threshold_mean(gray0))
plt.imshow(wiener(gray0))

#######################################
plt.imshow(img)
plt.imshow(gray0)
plt.imshow(image)
### TREES
plt.imshow(segmentation)
### CONTOURS
plt.imshow(img_back, cmap = 'gray')
### STREET
plt.imshow(gy)
plt.imshow(angle)
plt.imshow(binary_adaptive)
plt.imshow(binary_global)
#### STREET CALÇADA TREES BEST
io.imshow(filt_real)    
plt.imshow(gauss)
plt.imshow(lap)
plt.imshow(pre)
plt.imshow(pre_v)
#### CARS CONTOURS
plt.imshow(edges2)
plt.imshow(scharr(gray0))

plt.figure(figsize=(12,12))
ax = plt.subplot(1, 2, 1)
plt.imshow(img)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('ORIGINAL IMAGE')

ax = plt.subplot(1, 2, 2)
plt.imshow(gy)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('SOBEL FILTER')
plt.show()

plt.figure(figsize=(10,10))
ax = plt.subplot(2, 2, 1)
io.imshow(filt_real)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('GABOR FILTER')

ax = plt.subplot(2, 2, 2)
plt.imshow(edges2)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('ROBERTS FILTER')
plt.show()

