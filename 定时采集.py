
import time
import cv2
import os
def screencap():
    cap = cv2.VideoCapture(0) 
    c = 1
    if cap.isOpened():  
        rval, frame = cap.read()
    else:
        rval = False
    timeF = 300 
    start = time.time()
    while rval:
        rval, frame = cap.read()
        if (c % timeF == 0): 
            local_time = time.strftime("%Y%m%d%H%M%S", time.localtime()) 
            folder_name = time.strftime("%Y-%m-%d", time.localtime()) 
            file = '/home/mabo/Desktop/PvImages/{}/'.format(folder_name)
            if not os.path.exists(file):  
                os.makedirs(file)
            cv2.imwrite(file + str(local_time) + ".jpg", frame)
            end = time.time()
            print(str(end - start))
        c = c + 1
        cv2.waitKey(1)
    cap.release()

def stop_screencap():
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
	cv2.imshow('frame',frame)
        cv2.waitKey(50)



def main():
    local_time=time.strftime("%Y%m%d%H%M%S", time.localtime())
    hour_time=time.strftime("%H", time.localtime())
    if int(hour_time)>=4 and int(hour_time)<20:
	print(hour_time)
        screencap()
    if int(hour_time)>=20:
        stop_screencap()	
    if int(hour_time)<4:
        stop_screencap()

if __name__=="__main__":
    main()

