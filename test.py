### IMPORT OF LIBRARYS, ETC
import sys
import os.path
import cv2
import cv2.cv as cv
import numpy as np
from PIL import Image

#### DEFINE FUNCTIONS
def removeBackground(image):
	mask = image[...,1].copy();
	mask1 = image[...,2].copy();
	cv2.normalize(mask, mask, -30, 255,cv2.NORM_MINMAX)  ## NORMALIZE TO 256
	mask = (mask>220).choose(mask,0)
	cv2.normalize(mask1, mask1, -20, 255,cv2.NORM_MINMAX)
	mask1 = np.clip(mask1, 0, 1)
	mask = mask * mask1
	cv2.normalize(mask, mask, 0, 255,cv2.NORM_MINMAX)
	mask = np.uint8(np.around(mask))
	return mask


def detect_faces(image): # function that returns the location of a face in an submitted image
    imgArray = (cv.fromarray(image))
    HAAR_CASCADE_PATH = "C:\Python27\include\cbir\haarcascade_frontalface_default.xml"
    cascade = cv.Load(HAAR_CASCADE_PATH)
    storage = cv.CreateMemStorage()
    return cv.HaarDetectObjects(imgArray, cascade, cv.CreateMemStorage(0), 1.02, 1, cv.CV_HAAR_DO_CANNY_PRUNING, (20, 20))

	
### DEFINE VARIABLES

folder = 'C:\\wamp\\www\\womens-tops\\'
folder_out  = 'C:\\test\\tops\\'
######## PROGRAM START 


for filename in os.listdir (folder):
	#filename = "test.jpg"
	print folder+filename
	im = Image.open(folder+filename )
	imx = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2HLS)
	mask = removeBackground(imx)
	
	imgCrop = imx.copy()
	imgCrop[:,:,0] = 255-imx[:,:,0] * mask 
	imgCrop[:,:,1] = 255-imx[:,:,1] * mask
	imgCrop[:,:,2] = 255-imx[:,:,2] * mask
	
	ttt = imgCrop.min
	if ttt==255:
		continue
	#imgCrop = imgCrop[:,:,0:3] # Drop the alpha channel
	imgCrop = np.where(imgCrop -255)[0:2] # Drop the color when finding edgess
	
	box = map(min,imgCrop)[::-1] + map(max,imgCrop)[::-1]
	
	region = im.crop(box)

	img = cv2.cvtColor(np.asarray(region), cv2.COLOR_RGB2BGR)
	(z,z1,z2) = img.shape

	faces = detect_faces(img)
	for (x,y,w,h),n in faces:
		if (z>y+h*7 & z1> x+w*2):
			z=y+h*7
			z1=x+w*2
			cropped = img[y+h:z, x-w:z1,:].copy()
			cv2.imwrite(folder_out+str(n)+'_'+filename,cropped)
		
	


	###### PROGRAM END ###############		
				

#img = np.empty((65, 65), dtype=np.uint32) # (or dtype=np.uint16)
#print img.dtype.name
#imx = np.zeros((4161,4161,3),np.float32)

	
# for filename in os.listdir (folder):
	# flag = 0
	# im = cv2.imread(folder+'\\'+filename)
	# faces = detect_faces(im)
	# for (x,y,w,h),n in faces:
		# flag = 1
		# (yy,xx,zz) = img.shape
		# im = im[y:y+h,x:x+w,:].copy()
		# break
	
	# if flag==0:
		# continue
		
	# print folder+'\\'+filename
	# im = cv2.resize(im, (64,64))

	# img = np.zeros((64,64), np.float32)
	# img[:, :] = im[:,:,1]
    # #im=np.uint8(np.around(im))
	# #for i in range(0,8):
	# #	for j in range(0,8):
	# #		cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_FORWARD)
			# #cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_INVERSE)
	# cv2.dct(img,img, cv.CV_DXT_FORWARD)
	# x = 0
	# y = 0
	# counter = 0
	# total = 0
	# for i in range(0,64):
		# if i%2==0:
			# for j in range(0,i+1):
				# counter = counter +1
				# total = total + (img[i-j,j])/20
				# #print img[i-j,j]
				# #print (i-j),j
				# #img[i-j,j] = counter
				# imx[total,counter,0]+=10
		# else:
			# for j in range(0,i+1):
				# z=i-j
				# counter = counter +1
				# total = total + (img[i-z,z])/20
				# #img[i-z,z] = counter
				# imx[total,counter,0]+=10
				
	# for zz in range(0,64):
		# #zz = i;
		# if zz%2==0:
			# for j in range(0,zz+1):
				# counter = counter +1
				# total = total + (img[j-zz,63-j])/20
				# #img[j-zz,63-j] = counter
				# imx[total,counter,0]+=10
		# else:
			# for j in range(0,zz+1):
				# z=zz-j
				# counter = counter +1
				# total = total + (img[z-zz,63-z])/20
				# imx[total,counter,0]+=10
				# #img[z-zz,63-z] = counter
	# print counter
	# cv2.imwrite(folder+'\\'+(str)(total)+'_'+filename,im)	
	# #cv2.normalize(img,img,0,255*255,cv2.NORM_MINMAX)
	# #print np.amax(img)
	# #img=np.around(img)
	# #img = img.astype(np.int16)
	# ######### REVERSE ############
	# #img = img.astype(np.float32)
	

	# #for i in range(0,8):
	# #	for j in range(0,8):
			# #cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_FORWARD)
	# #		cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_INVERSE)
	
	# #img = img * 256;
	
# # folder = 'C:\\pants'
# # img = np.empty((65, 65), dtype=np.uint32) # (or dtype=np.uint16)
# # print img.dtype.name

# # for filename in os.listdir (folder):
	# # #print filename
	# # im = cv2.imread(folder+'\\'+filename)
	# # print folder+'\\'+filename
	# # im = cv2.resize(im, (64,64))
	# # #imx = np.zeros((4161,4161),np.float32)

	# # img = np.zeros((64,64), np.float32)
	# # img[:, :] = im[:,:,1]
    # # #im=np.uint8(np.around(im))
	# # #for i in range(0,8):
	# # #	for j in range(0,8):
	# # #		cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_FORWARD)
			# # #cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_INVERSE)
	# # cv2.dct(img,img, cv.CV_DXT_FORWARD)
	# # x = 0
	# # y = 0
	# # counter = 0
	# # total = 0
	# # for i in range(0,64):
		# # if i%2==0:
			# # for j in range(0,i+1):
				# # counter = counter +1
				# # total = total + (img[i-j,j])/20
				# # #print img[i-j,j]
				# # #print (i-j),j
				# # #img[i-j,j] = counter
				# # imx[total,counter,1]+=1000
		# # else:
			# # for j in range(0,i+1):
				# # z=i-j
				# # counter = counter +1
				# # total = total + (img[i-z,z])/20
				# # #img[i-z,z] = counter
				# # imx[total,counter,1]+=1000
				
	# # for zz in range(0,64):
		# # #zz = i;
		# # if zz%2==0:
			# # for j in range(0,zz+1):
				# # counter = counter +1
				# # total = total + (img[j-zz,63-j])/20
				# # #img[j-zz,63-j] = counter
				# # imx[total,counter,1]+=1000
		# # else:
			# # for j in range(0,zz+1):
				# # z=zz-j
				# # counter = counter +1
				# # total = total + (img[z-zz,63-z])/20
				# # imx[total,counter,1]+=1000
				# # #img[z-zz,63-z] = counter
	# # print counter			
	# # cv2.imwrite(folder+'\\'+(str)(total)+'_'+filename,img)
	# # #cv2.normalize(img,img,0,255*255,cv2.NORM_MINMAX)
	# # #print np.amax(img)
	# # #img=np.around(img)
	# # #img = img.astype(np.int16)
	# # ######### REVERSE ############
	# # #img = img.astype(np.float32)
	

	# # #for i in range(0,8):
	# # #	for j in range(0,8):
			# # #cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_FORWARD)
	# # #		cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_INVERSE)
	
	# # #img = img * 256;
	


# # folder = 'C:\\shoes'
# # img = np.empty((65, 65), dtype=np.uint32) # (or dtype=np.uint16)
# # print img.dtype.name

# # for filename in os.listdir (folder):
	# # #print filename
	# # im = cv2.imread(folder+'\\'+filename)
	# # print folder+'\\'+filename
	# # im = cv2.resize(im, (64,64))
	# # #imx = np.zeros((4161,4161),np.float32)

	# # img = np.zeros((64,64), np.float32)
	# # img[:, :] = im[:,:,1]
    # # #im=np.uint8(np.around(im))
	# # #for i in range(0,8):
	# # #	for j in range(0,8):
	# # #		cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_FORWARD)
			# # #cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_INVERSE)
	# # cv2.dct(img,img, cv.CV_DXT_FORWARD)
	# # x = 0
	# # y = 0
	# # counter = 0
	# # total = 0
	# # for i in range(0,64):
		# # if i%2==0:
			# # for j in range(0,i+1):
				# # counter = counter +1
				# # total = total + (img[i-j,j])/20
				# # #print img[i-j,j]
				# # #print (i-j),j
				# # #img[i-j,j] = counter
				# # imx[total,counter,2]+=1000
		# # else:
			# # for j in range(0,i+1):
				# # z=i-j
				# # counter = counter +1
				# # total = total + (img[i-z,z])/20
				# # #img[i-z,z] = counter
				# # imx[total,counter,2]+=1000
				
	# # for zz in range(0,64):
		# # #zz = i;
		# # if zz%2==0:
			# # for j in range(0,zz+1):
				# # counter = counter +1
				# # total = total + (img[j-zz,63-j])/20
				# # #img[j-zz,63-j] = counter
				# # imx[total,counter,2]+=1000
		# # else:
			# # for j in range(0,zz+1):
				# # z=zz-j
				# # counter = counter +1
				# # total = total + (img[z-zz,63-z])/20
				# # imx[total,counter,2]+=1000
				# # #img[z-zz,63-z] = counter
	# # print counter
	# # cv2.imwrite(folder+'\\'+(str)((total))+'_'+filename,im)
	# # #cv2.normalize(img,img,0,255*255,cv2.NORM_MINMAX)
	# # #print np.amax(img)
	# # #img=np.around(img)
	# # #img = img.astype(np.int16)
	# # ######### REVERSE ############
	# # #img = img.astype(np.float32)
	

	# # #for i in range(0,8):
	# # #	for j in range(0,8):
			# # #cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_FORWARD)
	# # #		cv2.dct(img[i*8:(i+1)*8,j*8:(j+1)*8], img[i*8:(i+1)*8,j*8:(j+1)*8], cv.CV_DXT_INVERSE)
	
	# # #img = img * 256;
	
	
# cv2.imwrite(folder+'\\i'+filename,imx)