import numpy as np
from skimage import img_as_ubyte		
from skimage.color import rgb2gray
import cv2
import datetime
import argparse
import imutils
import time
import torch
from time import sleep
from imutils.video import VideoStream
from CNN_NET import CNN_NET


path="/home/pi/Desktop/DIGIT RECOGNIZER/weights.h5"
model=torch.load(path)		

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)


def ImagePreProcess(im_orig, fr):

	im_gray = rgb2gray(im_orig)				
	img_gray_u8 = img_as_ubyte(im_gray)		
	
	(thresh, im_bw) = cv2.threshold(img_gray_u8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	img_resized = cv2.resize(im_bw,(28,28))
	
	im_gray_invert = 255 - img_resized ;
	im_final = im_gray_invert.reshape(1,1,28,28); im_final = torch.from_numpy(im_final);im_final = im_final.type('torch.FloatTensor')
	
	ans=model(im_final)
		
	ans = ans[0].tolist().index(max(ans[0].tolist())); a= "Predicted digit: "; b= str(ans); c=a+b; cv2.putText(fr, c, (70,270), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,225), 2);fr = imutils.resize(fr, width=400); cv2.imshow('OuTpUt',fr)
	
def main():

    t0=int(time.time());
    d=0
	
    while True:
        try:

            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            cv2.imshow("Show the digit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
                cv2.destroyAllWindows()
                vs.stop()
            else:
                cv2.imwrite("num.jpg", frame)
                im_orig = cv2.imread("num.jpg")
                ImagePreProcess(im_orig, frame)

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            vs.stop()

if __name__=="__main__":
	main()