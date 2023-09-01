# import libraries
import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt

# thresholds for various shape detection criteria
SPHERE_D_MAX = 100
SPHERE_D_MIN = 2
SPHERE_ROUNDNESS = 0.8

CUBE_L_MAX = 100
CUBE_L_MIN = 10
CUBE_SQUARENESS = 0.5

STAR_L_MAX = 100
STAR_L_MIN = 50
STAR_SQUARENESS = 0.8

ROD_W_MAX = 20
ROD_W_MIN = 5
ROD_L_MAX = 50
ROD_L_MIN = 10
ROD_AR_MIN = 1.5
ROD_AR_MAX = 10
ROD_SQUARENESS = 0.5

MAG = 100 #scale bar = 100 nm

IMG_R = 4
LABEL_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

def nothing(x):
    pass

# Function to detect the scale bar length in the image
def DetectScaleBar():
    # Code for detecting the scale bar length and returning it
    # img_sb = img[2050:2090,1200:2400] # scale bar image (height, width) (image WxH = 2468 x 2448)
    # img_sb = img[1475:1600,1155:1400] # scale bar image (height, width) (image WxH = 1920 x 1742) #don't use it
    img_sb = img[1450:1500,890:1600] # scale bar image (height, width) (image WxH = 1920 x 1742)
    cv2.imshow('bar', img_sb)
    # cv2.waitKey(0)

    ret,img_sb = cv2.threshold(img_sb,10,255,cv2.THRESH_BINARY_INV)
    bar,_ = cv2.findContours(img_sb,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for bari in bar:
        x,y,w,h = cv2.boundingRect(bari)
        # print (bari)
        area = cv2.contourArea(bari)
        if area > 0 and w * h / area <1.25 and w / h > 5:
            print (bari)
            return (w)
    return (0)

# Functions for detecting different shapes (sphere, cube, star, rod)
def FindSphere():
    center = (int(x),int(y))
    radius = int(r)
    if w > SPHERE_D_MIN and w < SPHERE_D_MAX \
       and area/(3.14*(w/2)**2) > SPHERE_ROUNDNESS: # standard
        cv2.circle(im,center,radius,(0,255,0),LABEL_THICKNESS)
        cv2.putText(im,str(npn + count),center,FONT,1,(0,255,0),LABEL_THICKNESS,cv2.LINE_AA)
        size_temp.append(w)
        return 1
    else:
        return 0


def FindCube():
    if w > CUBE_L_MIN and w < CUBE_L_MAX \
       and h/w < 1.3 and area/(w*h) > CUBE_SQUARENESS: # standard
        cv2.drawContours(im, [box], 0, (0,255,0), LABEL_THICKNESS)
        center = (int(rect[0][0]),int(rect[0][1]))
        cv2.putText(im,str(npn + count),center,FONT,1,(0,255,0),LABEL_THICKNESS,cv2.LINE_AA)
        size_temp.append((h + w)/2)
        return 1
    else:
        return 0

def FindStar():
    if w > STAR_L_MIN and w < STAR_L_MAX \
       and h/w < 1.2 and area/(w*h) < STAR_SQUARENESS: # standard
        cv2.drawContours(im, [box], 0, (0,255,0), LABEL_THICKNESS)
        center = (int(rect[0][0]),int(rect[0][1]))
        diagonal_length = np.sqrt(h**2 + w**2)
        # print(diagonal_length)
        cv2.putText(im,str(npn + count),center,FONT,1,(0,255,0),LABEL_THICKNESS,cv2.LINE_AA)
        size_temp.append((h,w,diagonal_length))
        return 1
    else:
        return 0

def FindRod():
    if w > ROD_W_MIN and w < ROD_W_MAX and h > ROD_L_MIN and h < ROD_L_MAX and \
        h/w > ROD_AR_MIN and h/w < ROD_AR_MAX and area/(w*h) > ROD_SQUARENESS: # standard
        cv2.drawContours(im, [box], 0, (0,255,0), LABEL_THICKNESS)
        center = (int(rect[0][0]),int(rect[0][1]))
        cv2.putText(im,str(npn + count),center,FONT,1,(0,255,0),LABEL_THICKNESS,cv2.LINE_AA)
        size_temp.append((h,w,h/w))
        return 1
    else:
        return 0

# Function for handling mouse events to capture an image region
def enlarge(event,x,y,flags,param): # inset image position
    global ix,iy,captureflag,x1,x2,y1,y2

    if event == cv2.EVENT_LBUTTONDOWN:
        captureflag = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if captureflag == True:
            x1 = min(ix,x)
            x2 = max(ix,x)
            y1 = min(iy,y)
            y2 = max(iy,y)

    elif event == cv2.EVENT_LBUTTONUP:
        captureflag = False

# Function to export size measurements to a file
def ExportSize():
    size_file = open(dirpath + '\\' + foldername + '\\results\\npsize.txt','w')
    if npn == 0:
        print ('No measurements!')
        size_file.write('No measurements! \n')
        size_file.close()
        return
    if nptype == 2:
        size_file.write("AVG_length=%s \nAVG_width=%s \nAVG_ar=%s \n" % \
                        (str(nph/npn),str(npw/npn),str(nph/npw)))
        std = np.std(size_all, axis=0)
        size_file.write("STD_length=%s \nSTD_width=%s \nSTD_ar=%s\n" % \
                        (str(std[0]), str(std[1]), str(std[2])))

        size_file.write("Index Length Width \n")
    if nptype == 3:
        size_file.write("AVG_length=%s \nAVG_width=%s \nAVG_diagonal=%s \n" % \
                        (str(nph/npn),str(npw/npn),str(npd/npn)))
        std = np.std(size_all, axis=0)
        size_file.write("STD_length=%s \nSTD_width=%s \nSTD_diagonal=%s\n" % \
                        (str(std[0]), str(std[1]), str(std[2])))

        size_file.write("Index Length Width Diagonal\n")
    else:
        size_file.write("AVG_length/diameter=%s \n\n" % \
                        (str((npw + nph)/2/npn)))
        std = np.std(size_all, axis=0)
        size_file.write("STD_diameter=%s \n\n" % \
                        (str(std)))
        size_file.write("Index Length \n")


    for i,size in enumerate(size_all):
        if nptype == 2:
            size_file.write("%s\t %s\t %s\t\n" % (i,size[0],size[1]))
        if nptype == 3:
            size_file.write("%s\t %s\t %s\t %s\t\n" % (i,size[0],size[1], size[2]))
        else:
            size_file.write("%s\t %s\t\n" % (i,size))

    size_file.close()

# Main program starts here
dirpath = os.getcwd()
nptype = -1 #nanoparticle type

for foldername in os.listdir(dirpath):
     # Detect the nanoparticle type based on the folder name and process images accordingly
    if foldername.startswith('sphere'):
        nptype = 0
    elif foldername.startswith('cube'):
        nptype = 1
    elif foldername.startswith('star'):
        nptype = 3
    elif foldername.startswith('rod'):
        nptype = 2
    else:
        continue
    size_all = []
    nph = 0
    npw = 0
    npn = 0
    npd = 0
    captureflag = False
    if os.path.isdir(foldername + '\\results') == False:
        os.mkdir(foldername + '\\results')
    for filename in os.listdir(dirpath + '\\' + foldername):
        # Iterate through image files in the nanoparticle's folder
        # Preprocess images, detect shapes, capture regions, and calculate size measurements
        if filename.endswith('.jpg') or filename.endswith('.tif'):
            img0 = cv2.imread(dirpath + '\\' + foldername + '\\' + filename)
            img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
            x1 = y1 = 0
            x2 = y2 = 1
            r,c = img.shape # 2464 * 2448
            # print (r,c)
            # raise
            r2 = int(r/IMG_R)
            c2 = int(c/IMG_R)
            mag = MAG / DetectScaleBar()
            #img_blur = cv2.GaussianBlur(img,(5,5),0)
            img_blur = cv2.medianBlur(img[0:1450,0:1742],5)
            # img_blur = cv2.medianBlur(img[0:2000,0:2488],5)

            #plt.hist(img_blur.ravel(),256,(0,255))
            # plt.show()

            cv2.namedWindow('dst')
            cv2.createTrackbar('Thresh','dst',50,100,nothing)
            cv2.setMouseCallback('dst',enlarge)
            threshflag = -1
            while(True):
                thresh = cv2.getTrackbarPos('Thresh','dst')
                if thresh != threshflag:
                    otsu,img_bw = cv2.threshold(img_blur,thresh,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    ret,img_bw = cv2.threshold(img_blur,otsu - 60 + thresh,255,cv2.THRESH_BINARY_INV)
                    #print (otsu)
                    cont,_= cv2.findContours(img_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    im = np.copy(img0)
                    width = 0
                    height = 0
                    diagonal_length = 0
                    count = 0
                    size_temp = []
                    for conti in cont:
                        #cv2.drawContours(im, [conti], 0, (0,0,255), 2)
                        rect = cv2.minAreaRect(conti)
                        area = cv2.contourArea(conti)*mag**2
                        box = np.int0(cv2.boxPoints(rect))
                        w = min(rect[1])*mag
                        h = max(rect[1])*mag
                        isnp = 0
                        if nptype == 0:
                            (x,y),r = cv2.minEnclosingCircle(conti)
                            h = w = 2*r*mag
                            isnp = FindSphere()
                        elif nptype == 1:
                            isnp = FindCube()
                        elif nptype == 2:
                            isnp = FindRod()
                        elif nptype == 3:
                            isnp = FindStar()
                        if isnp == 1:
                            width += w
                            height += h
                            diagonal_length += np.sqrt(h**2 + w**2)
                            count += 1
                    #if count > 0:
                            #print ('n=' + str(count) + '\n')
                    threshflag = thresh
                im_resize = cv2.resize(im,(c2,r2),interpolation = cv2.INTER_AREA)
                cv2.rectangle(im_resize,(x1,y1),(x2,y2),(0,0,255),1) # inset region
                cv2.imshow('dst',im_resize)
                if captureflag == False and y1 - y2 != 0 and x1 - x2 != 0:
                    im_inset = im[y1*IMG_R:y2*IMG_R,x1*IMG_R:x2*IMG_R]
                    cv2.imshow('inset',im_inset)
                k = cv2.waitKey(10) & 0xFF
                if k == ord('n'):
                    size_all.extend(size_temp)
                    cv2.imwrite(dirpath + '\\' + foldername + '\\results\\' + \
                                    os.path.splitext(filename)[0] + '_labelled.jpg',im)
                    npw += width
                    nph += height
                    npd += diagonal_length
                    npn += count
                    if npn == 0:
                        break
                    print ('Until now:')
                    if nptype == 2:
                        print ('w=' + str(npw/npn) + '\nh=' + str(nph/npn) + '\nnum=' + str(npn) + '\n')
                    if nptype == 3:
                        print ('w=' + str(npw/npn) + '\nh=' + str(nph/npn) + '\nD=' + str(npd/npn) + '\nnum=' + str(npn) + '\n')
                    else:
                        print ('l(/d)=' + str((npw + nph)/2/npn) + '\nnum=' + str(npn) + '\n')

                    cv2.destroyAllWindows()
                    break
                elif k == 27:
                    cv2.destroyAllWindows()
                    exit()
    ExportSize()

cv2.destroyAllWindows()
cv2.waitKey(0)
