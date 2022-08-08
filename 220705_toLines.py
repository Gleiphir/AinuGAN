# 参考
# https://emotionexplorer.blog.fc2.com/blog-entry-200.html

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 表示
def display_result_image(cap, color_image, skeleton):
    colorimg = color_image.copy()

    # カラー画像に細線化を合成
    colorimg = colorimg // 2 + 127
    colorimg[skeleton == 255] = 0

    cv2.imshow("",skeleton)
    cv2.imshow("",colorimg)
    cv2.waitKey(0)

# 細線化
def main():
    # 入力画像の取得
    colorimg = cv2.imread(r'D:\Github\AinuGAN\myDset-picked\H0011063-ima05-00-tp-org.png', cv2.IMREAD_COLOR)

    # グレースケール変換
    gray = cv2.cvtColor(colorimg, cv2.COLOR_BGR2GRAY)

    color_sobel_h = cv2.Sobel(colorimg, cv2.CV_8U, 0, 1, ksize=1)
    color_sobel_v = cv2.Sobel(colorimg, cv2.CV_8U, 1, 0, ksize=1)
    #_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3)
    #cv2.imshow(gray* 255)
    plt.imshow(color_sobel_h + color_sobel_v,cmap='gray')
    plt.show()
    # 二値画像反転
    image = cv2.bitwise_not(gray)

    # 細線化(スケルトン化) THINNING_ZHANGSUEN
    #skeleton1   =   cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    #display_result_image('ZHANGSUEN', colorimg, skeleton1)

    # 細線化(スケルトン化) THINNING_GUOHALL
    #skeleton2   =   cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    #display_result_image('GUOHALL', colorimg, skeleton2)

if __name__ == '__main__':
    main()