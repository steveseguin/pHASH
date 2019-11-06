### IMPORT OF LIBRARYS, ETC
import sys
import os.path
import cv2
import cv2.cv as cv
import numpy as np


folder = 'C:\\wamp\\www\\womens-tops\\'
filename = "243.jpg"
print folder+filename

img = cv2.imread(folder+filename)

img = cv2.cvtColor(img,  cv.CV_RGB2YCrCb);
img = cv2.split(img)[0] 

cv2.blur(img, (7,7), img, (1,1), cv2.BORDER_DEFAULT) 

img =cv2.resize(img, (32,32), 0, 0, cv2.INTER_AREA)

im = np.zeros((32,32), np.float32) # dumb way of converting to float32
im[:,:] = img[:,:]
im = cv2.dct(im, cv.CV_DXT_FORWARD) #DCT

im = im[0:8,0:8]  # crop 8x8 top left
cv2.imwrite("steves.png", im)

one = 0x000000000000000000000000000000001
hashval = 0x000000000000000000000000000000000
im = im.flatten()[:132]
median = np.median(im)

# Create hash
for ele in im:
	if (ele > median):
		hashval |= one	
	one = one << 1

hashval = "%16x" % hashval
print hashval


