#queryTree(char* treename, float rad, int kn, char*name, char *hash){

#import pHash
import Image, cv, os, sys, numpy, argparse
import PIL.Image as pil
from urllib2 import urlopen
from  cStringIO import StringIO
from sys import argv, stdout
import cv2
import cv2.cv as cv
import numpy as np
from copy import deepcopy 
from math import pow
from cv import *
import glob
import ctypes

#Ctypes stuff
def main():
	# # load the shared object
	# libtest = ctypes.cdll.LoadLibrary('./imget.so')
	# #Calling C function to add to MVP tree 
	# def call_c(filename, hashval):
	    # libtest.queryTree("trytreenew", ctypes.c_float(40), ctypes.c_int(100), ctypes.c_char_p(filename), ctypes.c_char_p(hashval))



	#Get image and open
	args = str(argv[1])

	#Getting filename
	filename = os.path.basename(args)
	print (args)
	
	im = pil.open(args).convert("RGB") 
	imgcv = cv2.cvtColor(np.asarray(im), cv.CV_RGB2YCrCb)

	#Get the channel 0 
	imgcv = cv2.split(imgcv)[0] 
	#cv2.imwrite("1Ychannelimage.jpg", imgcv) 
	imgcva = imgcv
	
	# Improve box filter
	imgcva = cv2.boxFilter(imgcv, -1, (7,7), imgcva, (-1,-1), True, cv2.BORDER_DEFAULT)
	#cv2.imwrite("2AfterBoxFilter.jpg", imgcva) 
	
	#Correct resize stuff
	pil_im = Image.fromarray(np.uint8(imgcva))
	imgcva = numpy.array(pil_im.resize((32, 32)))
	#cv2.imwrite("3AfterResizing.jpg", imgcva) 
	
	#DCT
	imf = np.float32(imgcva)  # float conversion/scale
	dst = cv2.dct(imf)           # the dct
	imgcv = np.uint8(dst) # convert back
	#cv2.imwrite("4AfterDCT.jpg", imgcv) 

	#Crop 8 x 8
	cropped = cv.CreateImage((8, 8), cv.IPL_DEPTH_8U, 1)
	src_region = cv.GetSubRect(cv.fromarray(imgcv), (1, 1, 8, 8) )
	
	

	cv.Copy(src_region, cropped)
	hell = (np.array(src_region))

	cv2.imwrite("sho.png", hell) 

	#get to a string
	one = 0x000000000000000000000000000000001
	hashval = 0x000000000000000000000000000000000
	hell1 = hell.flatten()[:132]
	median = numpy.median(hell1)

	# Create hash
	for ele in hell1:
		if (ele > median):
			hashval |= one	
		one = one << 1
	print hashval
	hashval = "%16x" % hashval
	print hashval
	
main()
	