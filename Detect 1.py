# import the necessary packages
import cv2
import imutils
#load the face detector
detector_face = cv2.CascadeClassifier("cascades\\haarcascade_frontalface_default.xml")
#load the eye detector
detector_eye = cv2.CascadeClassifier("cascades\\haarcascade_eye.xml")
#Get the camera
camera= cv2.VideoCapture(0)

#We go on a continous capture from webcam
while True:
	#grab the current frame
	(grabbed,frame) = camera.read()
	#if no frame is grabed then break the loop
	if not grabbed:
		break
	#Resize the frame for better speed,convert it to grayscale,then detect faces and eyes
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#Detect Faces and get the Boundling Boxes
	faceRects = detector_face.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
	#loop over the faces and draw a rectangle around each
	for (x,y,w,h) in faceRects:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
	#Detect Eyes and get the Bounding Boxes
	eyeRects= detector_eye.detectMultiScale(gray)
	for(ex,ey,ew,eh) in eyeRects:
		#loop over the eyes and draw a rectangle around each
		cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	#show the frame to our screen
	cv2.imshow("Frame",frame)
	key = cv2.waitKey(1) & 0xFF
	#if the 'g' key is pressed,stop the loop
	if key == ord("q"):
		break
#cleanup the camera and close any open windows
camera.release()
cv2.destroyALLWindows()



