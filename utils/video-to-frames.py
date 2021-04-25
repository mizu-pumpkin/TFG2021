import cv2
import os

#os.chdir('data/drink')

for filename in os.listdir('.'):
    videoid = os.path.splitext(filename)[0] # remove video extension
    if os.path.isdir(videoid) == False: # check if directory exists
        os.mkdir(videoid)

    # Opens the Video file
    cap = cv2.VideoCapture(videoid+'.mp4')
    i=0
    while(cap.isOpened()):
        framename = os.path.join(videoid, videoid+'_'+str(i)+'.jpg')
        if os.path.isfile(framename) == True:
            break
        ret, frame = cap.read()
        if ret == False:
            break
        print('saving...',framename,frame.shape)
        cv2.imwrite(framename, frame)
        i+=1
    # Closes the Video file
    cap.release()

cv2.destroyAllWindows()