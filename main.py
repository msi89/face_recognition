import cv2
import dlib # http://dlib.net/python/index.html
import PIL.Image
import numpy as np 
from imutils import face_utils # https://github.com/PyImageSearch/imutils




# https://github.com/davisking/dlib-models
pose_predictor_68_point = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()


# Для получения координаты лица 
def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


def encode_face(image):
    face_locations = face_detector(image, 1)
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # разпразвание лицо
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list


def recognize_face(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    # ЛИЦО ЭНКОДИРОВАНИЕ
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # ПРОВЕРКА РАССТОЯНИЯ МЕЖДУ ИЗВЕСТНЫМИ ЛИЦАМИ И ОБНАРУЖЕННЫМИ ЛИЦАМИ
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Unknown"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)


if __name__ == '__main__':
    # импорт изображения
    face_to_encode_path = ["faces/me.jpg", "faces/mzuk.png", "faces/musk.webp"]
    known_face_names = ["Amon", "Mark Zuckerberg", "Elon Musk"]

    known_face_encodings = []
    for file_ in face_to_encode_path:
        image = PIL.Image.open(file_)
        image = np.array(image)
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)

    # запупскает видео камеры
    video_capture = cv2.VideoCapture(0)
  
    while True:
        
        ret, frame = video_capture.read()
        recognize_face(frame, known_face_encodings, known_face_names)
        cv2.imshow('Recognition App', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    # Остановить видео самары
    video_capture.release()
    cv2.destroyAllWindows()
