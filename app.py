import cv2
import deepface.DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference = cv2.imread("ref.jpg")
distance = 0
age = 0
dominant_gender = "Unknown"
dominant_emotion = "Unknown"


def check_face(frame):

    global face_match
    verification = deepface.DeepFace.verify(frame, reference)
    distance = verification["distance"]

    try:
        if verification["verified"]:
            face_match = True

        else:
            face_match = False

    except ValueError:
        face_match = False

    return face_match, distance


def analyze_face(frame):

    analysis = deepface.DeepFace.analyze(frame)
    age = analysis[0]["age"]
    dominant_gender = analysis[0]["dominant_gender"]
    dominant_emotion = analysis[0]["dominant_emotion"]

    return age, dominant_gender, dominant_emotion


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                face_match, distance = check_face(frame)
                age, dominant_gender, dominant_emotion = analyze_face(frame)
            except ValueError:
                pass

        counter += 1
        if face_match:
            cv2.putText(frame, "Match!", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 0))

        else:
            cv2.putText(frame, "Mismatch!", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255))

        cv2.putText(frame, f"{distance}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 0))
        cv2.putText(frame, f"{age}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0))
        cv2.putText(frame, f"{dominant_gender}", (220, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0))
        cv2.putText(frame, f"{dominant_emotion}", (420, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0))

        cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
