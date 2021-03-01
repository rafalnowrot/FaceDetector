import cv2
import sys

cascPath = sys.argv[0]
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (250, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        crop_img = frame[y: y + h, x: x + w] 
        cv2.imwrite("face.jpg", crop_img)

    count = 0

    for (x,y,w,h) in faces:
            face = frame[y-40:y+h+40, x-40:x+w+40] #slice the face from the image
            cv2.imwrite(str(count)+'.jpg', face) #save the image
            count+=1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


