# FFT ALGORITHM
import cv2
import numpy as np

image = cv2.imread('G:\My Files\Sem-8\CSE-445\h.png', cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(image)
f_shift = np.fft.fftshift(f)

mag_spec = 20*np.log(np.abs(f_shift))
mag_spec = np.asarray(mag_spec, dtype=np.uint8)
image_and_magnitude = np.concatenate((image,mag_spec),axis=1)

#cv2.imshow('magnitude spectrum', mag_spec)
cv2.imshow('Image', image_and_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()


# HAAR CASCADES FILTER
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye_default.xml')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

catface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    roi = img[50:250,200:400]
    cv2.imshow('ROI', roi)
    cv2.moveWindow('ROI', 705, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
    '''
    eyes = eye_cascade.detectMultiScale(roi_gray)    
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh), (0,255,0), 2)
    '''        
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows




