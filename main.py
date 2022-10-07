#  USE 'Q' PARA SAIR!!


import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()



#resize video
def resize_src(percentage, image, width, height):
    result = cv2.resize(image, (int(width * percentage),int(height * percentage)), interpolation= cv2.INTER_LINEAR)
    return result
def place_points(rgb_image, width, height):
    result = face_mesh.process(rgb_image)

    for facial_landmark in result.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmark.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)

            cv2.circle(image,(x,y), 2, (200, 10, 10), -1)

def init_video_set(image):
    image = cv2.flip(image, 0)
    height, width, _ = image.shape
    percentage = 0.5
    image = resize_src(percentage, image, width, height)

    return image

cap = cv2.VideoCapture("resources/cicero.mp4")

while(cap.isOpened()):
    ret, image = cap.read()
    image = init_video_set(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    s = image.shape

    if ret == True:
        place_points(rgb_image, s[1], s[0])
        cv2.imshow('frame', image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()