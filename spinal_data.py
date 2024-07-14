# -*- coding: utf-8 -*- -#
import cv2
import numpy as np
import matplotlib as mat
import glob
import os

#파일 불러오기
image_dir = "bw_ok"
print(image_dir)
files = glob.glob(image_dir + "/*.bmp")  # 파일 이름 다불러옴
print(files)
for i, f in enumerate(files):
   # img = Image.open(f)
    origin = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(origin, (380, 380))
    equalizeimage = cv2.equalizeHist(image)
    dst = equalizeimage.copy()
    #cv2.imshow("IMAGE"+str(i),equalizeimage)
    cv2.waitKey(100)

w = equalizeimage.shape[1]
h = equalizeimage.shape[0]

def rectify(img, w, h):
    for i in range (0, h):
        for j in range(0, w) :
            if img[i][j]<0.0 :
                img[i][j]=0.0
            elif img[i][j]>255.0 :
                img[i][j]=255.0

def sigmoid_transform(a, c, LUT):
    i = 0
    for i in range(i, 256):
        s = i/255.0
        y = np.exp(-a*(s-c))
        tmp = 255.0 *(1/(1+y))
        if tmp > 255:
            tmp = 255
        LUT[i] = tmp

def nothing(x):
    pass

def processing(img,k):
    LUT = np.zeros(256, np.uint8)
    f_img = np.float32(img)
    smooth5 = cv2.GaussianBlur(f_img,(5,5), 0,0)
    smooth11 = cv2.GaussianBlur(f_img,(11,11), 0,0)
    smooth17 = cv2.GaussianBlur(f_img,(17,17), 0,0)
    smooth23 = cv2.GaussianBlur(f_img,(23,23), 0,0)
    smooth29 = cv2.GaussianBlur(f_img,(29,29), 0,0)
    smooth35 = cv2.GaussianBlur(f_img,(35,35), 0,0)

    diff = img - smooth5
    diff1 = smooth5 - smooth11
    diff2 = smooth11 - smooth17
    diff3 = smooth17 - smooth23
    diff4 = smooth23 - smooth29
    diff5 = smooth29 - smooth35

    # enhance 영상
    sharp1 = img + 2 * diff
    sharp2 = img+ 2 * diff1
    sharp3 = img + 2 * diff2
    sharp4 = img + 2 * diff3
    sharp5 = img + 2 * diff4
    sharp6 = img + 2 * diff5

    sharp = (img + sharp1 + sharp2 + sharp3 + sharp4 + sharp5 + sharp6 + smooth35 * (-4.0)) * 0.23  # + sharp7 + smooth35*(-4.0))*0.23
    rectify(sharp, w, h)
    view = np.uint8(sharp)
    #cv2.imshow("sharp", view)

    sigmoid_transform(16, 0.69, LUT)  # 영상 대비, 영상 필터링
    dst = np.take(LUT, np.uint8(sharp))
    view2 = np.uint8(dst)

    none = np.average(img[w // 2:w])  # 아래 반절 사진
    av_w = np.average(img[w // 2:w]) * 0.3
    # cv2.imshow("half", equalizeimage[w//2:w]) #특정 범위 내의 사진만 보기
    print(none)
    print(av_w)

    for i in range(w // 2, w):
        for j in range(h):
            if view2[i][j] < av_w:
                view2[i][j] = 0
            else:
                view2[i][j] = view2[i][j]
    cv2.imshow("result",view2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    path = "Processing_normal"
    cv2.imwrite(os.path.join(path, str(k) + "ResultBW1.bmp"), view2)

for i, f in enumerate(files):
   # img = Image.open(f)
    origin = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(origin, (380, 380))
    equalizeimage = cv2.equalizeHist(image)
    dst = equalizeimage.copy()
    processing(equalizeimage,str(i))