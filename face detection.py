import cv2

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
imp_img = cv2.VideoCapture("vlcsnap-2021-09-24-10h09m50s198.png")

res, img = imp_img.read()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = detect.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img,(x,y), (x+w, y+h), (255,255,0), 2)

cv2.imshow("Ryan Reynold Image", img)
cv2.waitKey(60000)
imp_img.release()
cv2.destroyAllWindows()
