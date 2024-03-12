import threading

import cv2
import deepface.DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)



counter=0
face_match=False
reference=cv2.imread("ref.jpg")

def checkFace():
    global face_match
    global distance
    distance=deepface.DeepFace.verify(frame,reference)["distance"]
    try:
        if deepface.DeepFace.verify(frame, reference)["verified"]:
            face_match=True

        else:
            face_match=False


    except ValueError:
        face_match=False

while True:
    ret, frame=cap.read()

    if ret:
        if counter%30==0:
            try:
                threading.Thread(target=checkFace(), args=(frame.copy(),)).start()
            except ValueError:
                pass

        counter+=1
        if face_match:
            cv2.putText(frame, "Match!", (20,450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0,255,0))
            cv2.putText(frame, f"{distance}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 0))

        else:
            cv2.putText(frame, "Mismatch!", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255))
            cv2.putText(frame, f"{distance}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255))


        cv2.imshow("video",frame)
    key=cv2.waitKey(1)
    if key==ord("q"):
        break

cv2.destroyAllWindows()
