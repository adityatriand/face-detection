import cv2 as cv
import numpy as np
cascade_face_default = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_face_alt = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
if cascade_face_default.empty() or cascade_face_alt.empty() :
    raise IOError('File XML tidak bisa ditemukan')
foto = cv.imread('foto-example.jpg')
foto_gray = cv.cvtColor(foto, cv.COLOR_BGR2GRAY)
result_alt = cascade_face_alt.detectMultiScale(foto_gray, scaleFactor=1.02, minNeighbors=7)
result_default= cascade_face_default.detectMultiScale(foto_gray, scaleFactor=1.02, minNeighbors=7)
print("Terdeteksi ", len(result_alt), " orang di dalam gambar dengan haarcascade alt")
print("Terdeteksi ", len(result_default), " orang di dalam gambar dengan haarcascade default")
for (x,y,w,h) in result_alt:
    cv.rectangle(foto, (x,y), (x+w,y+h), color=(0,255,0), thickness=3)
cv.imwrite('result-detection-alt.jpg', foto)
for (x, y, w, h) in result_default:
    cv.rectangle(foto, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
cv.imwrite('result-detection-default.jpg', foto)

scale_percent = 60  # percent of original size
width = int(foto.shape[1] * scale_percent / 100)
height = int(foto.shape[0] * scale_percent / 100)
dim = (width, height)

img1 = cv.imread('result-detection-alt.jpg')
resized1 = cv.resize(img1, dim, interpolation=cv.INTER_AREA)

img2 = cv.imread('result-detection-default.jpg')
resized2 = cv.resize(img2, dim, interpolation=cv.INTER_AREA)

result = np.concatenate((resized1, resized2), axis=0)
cv.imshow('Face Detection', result)
cv.waitKey(0)
cv.destroyAllWindows()