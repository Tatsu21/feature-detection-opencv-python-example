# Project: Feature detection on video


import os
import shutil
import cv2
from configparser import ConfigParser
nrpoz= 1
parser = ConfigParser()
parser.read('config.ini')

n = 1
videos = {}
images = {}
durate = {}
avgorb=[]



def calculavg(list):
    sum = 0
    for i in list:
        sum = sum + i
    avg = sum/len(list)
    return avg

def ordiner():
    for i, x in parser.items('VIDEOS'):
        n = 1
        for c in x.split(","):
            if n == 1:
                videos[int(i)] = c
            if n == 2:
                images[int(i)] = c
            elif n == 3:
                durate[int(i)] = int(c)
            n += 1

def bfmatcher(img,img2,des1,des2,kp1,kp2,distance):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    percent=0.0
    for m, n in matches:
        if m.distance < distance * n.distance:
            good.append([m])
            a = len(good)
            percent = (a * 100) / len(kp2)
            print("{} % similarity".format(percent))
            if percent >= 75.00:
                print('Match')
            if percent < 75.00:
                print('finding..')
    avgorb.append(len(good))
    msg1 = 'There is %d similarity' % (len(good))
    msg2 = 'Avg good %f'%(calculavg(avgorb))
    msg3 = "{} %".format(int(percent))
    match_img = cv2.drawMatchesKnn(img, kp1, img2, kp2, good, None, flags=0)
    cv2.putText(match_img, msg1, (int(parser.get('VIDEO','height')), 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 180)
    cv2.putText(match_img, msg2, (int(parser.get('VIDEO', 'height')), 284), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 180)
    cv2.putText(match_img, msg3, (int(parser.get('VIDEO', 'height')), 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 180)
    cv2.imshow("Screen", match_img)
    return good,avgorb,match_img,percent

def akaze(img, img2,name,distance=0.81, vxdim=600, vydim=600, threshold=0.01579,nOctaves=2,nOctaveLayers=11):
    kaze = cv2.AKAZE_create(threshold=threshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers)
    img = cv2.resize(img, (vxdim, vydim))
    kp1, des1 = kaze.detectAndCompute(img, None)
    kp2, des2 = kaze.detectAndCompute(img2, None)
    good, avgorb,imgf,percent = bfmatcher(img, img2, des1, des2, kp1, kp2,distance)
    if(percent >= 90):
        path = "extract/"+str(name)+"/AKAZE/img"+str(nrpoz)+".png"
        cv2.imwrite(path,imgf)
    return len(good), calculavg(avgorb),percent

def sift(img, img2,name,distance=0.88,vxdim=600, vydim=600, nfeatures=106,contrastThreshold=0.18,edgeThreshold=13,sigma=3.4):
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold,sigma=sigma)
    img = cv2.resize(img, (vxdim, vydim))
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    good, avgorb,imgf,percent = bfmatcher(img, img2, des1, des2, kp1, kp2,distance)
    if(percent >= 90):
        path = "extract/"+str(name)+"/SIFT/img"+str(nrpoz)+".png"
        cv2.imwrite(path,imgf)
    return len(good), calculavg(avgorb),percent

def kaze(img,img2,name,distance=0.86,vxdim=600, vydim=600,extended=False,upright=True,threshold=0.01171,nOctaves=2,nOctaveLayers=11):
    kaze = cv2.KAZE_create(extended=extended,upright=upright, threshold=threshold, nOctaves=nOctaves,nOctaveLayers=nOctaveLayers)
    img = cv2.resize(img, (vxdim, vydim))
    kp1, des1 = kaze.detectAndCompute(img, None)
    kp2, des2 = kaze.detectAndCompute(img2, None)
    good, avgorb, imgf, percent = bfmatcher(img, img2, des1, des2, kp1, kp2, distance)
    if(percent >= 90):
        path = "extract/"+str(name)+"/KAZE/img"+str(nrpoz)+".png"
        cv2.imwrite(path,imgf)
    return len(good), calculavg(avgorb),percent

def brisk(img, img2,name, distance=0.88,vxdim=600, vydim=600, thresh=150, octaves=3,patternScale=3.8):
    brisk = cv2.BRISK_create(thresh=thresh, octaves=octaves, patternScale=patternScale)
    img = cv2.resize(img, (vxdim, vydim))
    kp1, des1 = brisk.detectAndCompute(img, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)
    good, avgorb,imgf,percent = bfmatcher(img, img2, des1, des2, kp1, kp2,distance)
    if(percent >= 90):
        path = "extract/"+str(name)+"/BRISK/img"+str(nrpoz)+".png"
        cv2.imwrite(path,imgf)
    return len(good), calculavg(avgorb),percent

def ORB(img, img2,name, distance=0.97,vxdim=600, vydim=600, nfeatures = 100, scaleFactor=1.91, nlevels=10, WTA_K=4):
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, WTA_K=WTA_K)
    img = cv2.resize(img, (vxdim, vydim))
    kp1, des1 = orb.detectAndCompute(img, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    good, avgorb,imgf,percent = bfmatcher(img, img2, des1, des2, kp1, kp2,distance)
    if(percent >= 90):
        path = "extract/"+str(name)+"/ORB/img"+str(nrpoz)+".png"
        cv2.imwrite(path,imgf)
    return len(good), calculavg(avgorb),percent

nr=0
shutil.rmtree('extract',ignore_errors=True)
shutil.rmtree('txt',ignore_errors=True)

while True:
    ordiner()

    cap = cv2.VideoCapture('video/'+str(videos.get(n)))
    img = cv2.imread('img/'+str(images.get(n)))

    if(durate.get(n) != 1):
        durata = durate.get(n)
    else:
        durata = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    nrpoz=0
    path1= "extract/"+str(videos.get(n))+"/SIFT"
    path2= "extract/" + str(videos.get(n)) + "/BRISK"
    path3= "extract/" + str(videos.get(n)) + "/ORB"
    path4= "extract/" + str(videos.get(n)) + "/AKAZE"
    path5= "extract/" + str(videos.get(n)) + "/KAZE"
    txt="txt/"+str(videos.get(n))

    try:
        os.makedirs(path1)
        os.makedirs(path2)
        os.makedirs(path3)
        os.makedirs(path4)
        os.makedirs(path5)
        os.makedirs(txt)
    except OSError:
        print("This directory wan`t created: %s" % path1)
        print("This directory wan`t created: %s" % path2)
        print("This directory wan`t created: %s" % path3)
        print("This directory wan`t created: %s" % path4)
        print("This directory wan`t created: %s" % path5)
        print("This directory wan`t created: %s" % txt)
    else:
        print("Directory %s was created" % path1)
        print("Directory %s was created" % path2)
        print("Directory %s was created" % path3)
        print("Directory %s was created" % path4)
        print("Directory %s was created" % path5)
        print("Directory %s was created" % path5)

    fileformatsift = txt+"/"+str(videos.get(n))+'_sift.txt'
    txtsift = open(fileformatsift,'w')
    fileformatsift = txt+"/"+str(videos.get(n))+'_brisk.txt'
    txtbrisk = open(fileformatsift,'w')
    fileformatsift = txt+"/"+str(videos.get(n))+'_ORB.txt'
    txtORB = open(fileformatsift,'w')
    fileformatsift = txt+"/"+str(videos.get(n))+'_akaze.txt'
    txtakaze = open(fileformatsift,'w')
    fileformatsift = txt+"/"+str(videos.get(n))+'_kaze.txt'
    txtkaze = open(fileformatsift,'w')
    for zx in range(5):
        for i in range(int(durata)):
            ret, frame = cap.read()
            nr+=1
            nrpoz+=1
            img = cv2.resize(img,(int(parser.get('VIDEO','pheight')),int(parser.get('VIDEO','pwidth'))))
            if(zx==1):
                if(parser.get('SIFT', 'configurable')=='True'):
                    x,y,z =sift(frame,img,videos.get(n),distance=float(parser.get('SIFT','distance')),vxdim=int(parser.get('VIDEO','height')), vydim=int(parser.get('VIDEO','width')), nfeatures=int(parser.get('SIFT', 'nfeatures')),contrastThreshold=float(parser.get('SIFT', 'contrastThreshold')),edgeThreshold=int(parser.get('SIFT', 'edgeThreshold')),sigma=float(parser.get('SIFT', 'sigma')))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtsift.write(format)
                else:
                    x,y,z =sift(frame,img,videos.get(n))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtsift.write(format)
            elif(zx==0):
                if(parser.get('BRISK', 'configurable') == 'True'):
                    x,y,z =brisk(frame,img,videos.get(n),distance=float(parser.get('BRISK','distance')),vxdim=int(parser.get('VIDEO','height')), vydim=int(parser.get('VIDEO','width')), thresh=int(parser.get('BRISK','thresh')), octaves=int(parser.get('BRISK','octaves')),patternScale=float(parser.get('BRISK','patternScale')))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtbrisk.write(format)
                else:
                    x,y,z =brisk(frame,img,videos.get(n))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtbrisk.write(format)
            elif(zx==2):
                if(parser.get('ORB', 'configurable') == 'True'):
                    x,y,z =ORB(frame,img,videos.get(n),distance=float(parser.get('ORB','distance')),vxdim=int(parser.get('VIDEO','height')),vydim=int(parser.get('VIDEO','width')),nfeatures=int(parser.get('ORB','nfeatures')),scaleFactor=float(parser.get('ORB','scaleFactor')),nlevels=int(parser.get('ORB','nlevels')))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtORB.write(format)
                else:
                    x,y,z =ORB(frame,img,videos.get(n))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtORB.write(format)
            elif(zx==3):
                if(parser.get('AKAZE', 'configurable')=='True'):
                    x,y,z =akaze(frame,img,videos.get(n),distance=float(parser.get('AKAZE','distance')),vxdim=int(parser.get('VIDEO','height')), vydim=int(parser.get('AVIDEO','width')), threshold=float(parser.get('AKAZE','threshold')),nOctaves=int(parser.get('AKAZE','nOctaves')),nOctaveLayers=int(parser.get('AKAZE','nOctaveLayers')))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtakaze.write(format)
                else:
                    x,y,z =akaze(frame,img,videos.get(n))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtakaze.write(format)
            elif(zx==4):
                if(parser.get('KAZE', 'configurable')=='True'):
                    x,y,z =kaze(frame,img,videos.get(n),distance=float(parser.get('KAZE','distance')),vxdim=int(parser.get('VIDEO','height')), vydim=int(parser.get('AVIDEO','width')),extended=bool(parser.get('KAZE','extended')),upright=bool(parser.get('KAZE','upright')),threshold=float(parser.get('KAZE','threshold')),nOctaves=int(parser.get('KAZE','nOctaves')),nOctaveLayers=int(parser.get('KAZE','nOctaveLayers')))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtkaze.write(format)
                else:
                    x,y,z =kaze(frame,img,videos.get(n))
                    format = str(nr)+' '+str(x) + ' ' + str(y)+' '+str(z)+'\n'
                    txtkaze.write(format)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                txtbrisk.close()
                txtsift.close()
                txtkaze.close()
                txtakaze.close()
                break
        cap.release()
        nr = 0

        cap = cv2.VideoCapture('video/' + str(videos.get(n)))


    txtbrisk.close()
    txtsift.close()
    txtkaze.close()
    txtakaze.close()
    avgorb.clear()
    n +=1
    cap.release()
    if len(videos)+1 == n:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()