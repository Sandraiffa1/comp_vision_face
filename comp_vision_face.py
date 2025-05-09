import cv2
import numpy as np
import mediapipe as mp
import gradio as gr
import json

# Инициализация MediaPipe для детекции лиц и сетки лиц
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


# проверка на тени: сделать чб обработку, проверять количество черных участков
#  - соот-но либо много теней, либо мало
# отделить фон: как будто не обязательно

def segment_image(image):
    blur_image = cv2.blur(image, (5, 5))
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(blur_image)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                return bbox
    return None  # Если лицо не найдено


# Классификация по уровню шума
# def classify_noise(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     noise_level = np.var(gray)
#     if noise_level < 200:
#         return "Low"
#     elif 200 <= noise_level < 2000:
#         return "Medium"
#     else:
#         return "High"
def classify_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    diff = cv2.absdiff(gray, blurred)
    noise_level = np.mean(diff)
    if noise_level < 3:
        return "Low"
    elif noise_level < 10:
        return "Medium"
    else:
        return "High"


# Классификация по контрастности
# def classify_contrast(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     contrast = gray.max() - gray.min()
#     if contrast < 50:
#         return "Low"
#     elif contrast < 150:
#         return "Medium"
#     else:
#         return "High"
def classify_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    if std < 50:
        return "Low"
    elif std < 80:
        return "Medium"
    else:
        return "High"


# Проверка размытости фона
def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return "Blurred" if blur < 100 else "Sharp"


# измнеила определение цвета
# Проверка цвета фона
COLOR_NAMES = {
    (0, 0, 0): "Black",
    (255, 255, 255): "White",
    (255, 0, 0): "Red",
    (0, 255, 0): "Lime",
    (0, 0, 255): "Blue",
    (255, 255, 0): "Yellow",
    (0, 255, 255): "Cyan",
    (255, 0, 255): "Magenta",
    (192, 192, 192): "Silver",
    (128, 128, 128): "Gray",
    (128, 0, 0): "Maroon",
    (128, 128, 0): "Olive",
    (0, 128, 0): "Green",
    (128, 0, 128): "Purple",
    (0, 128, 128): "Teal",
    (0, 0, 128): "Navy",
}


def closest_color_name(rgb_color):
    r, g, b = rgb_color
    min_distance = float("inf")
    closest_name = "Unknown"
    for color_rgb, name in COLOR_NAMES.items():
        cr, cg, cb = color_rgb
        distance = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name


def check_background_color(image):
    avg_color_bgr = np.mean(image, axis=(0, 1))
    avg_color_rgb = tuple(int(c) for c in avg_color_bgr[::-1])  # BGR → RGB
    color_name = closest_color_name(avg_color_rgb)
    return f"{color_name} (RGB: {avg_color_rgb})"


# изменила порог освещенности
# Проверка освещенности
def check_illumination(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    return "Uneven Light" if brightness < 180 else "Even Light"


# выразила в процентах
# !!сделать так же низко/средке/высоко но через проценты
# Проверка яркости
def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    brightness_pr = (brightness / 255) * 100
    return round(brightness_pr, 2)


# Анализ размера объекта и положения
def analyze_size(bbox, image_shape):
    h, w, _ = image_shape
    size_ratio = (bbox[2] * bbox[3]) / (h * w)  # Процентное представление размера лица
    face_size_perc = round(size_ratio * 100, 2)
    if size_ratio < 0.01:
        size_category = "Small"
    elif 0.01 <= size_ratio < 0.05:
        size_category = "Medium"
    else:
        size_category = "Large"

    return size_category, face_size_perc


# добавлено отклонение от центра
def calculate_face_center_offset(image):
    bbox = segment_image(image)
    if bbox is None:
        return None, None  # Лицо не найдено

    h, w, _ = image.shape
    img_center_x = w / 2
    img_center_y = h / 2

    face_center_x = bbox[0] + bbox[2] / 2
    face_center_y = bbox[1] + bbox[3] / 2

    offset_x = ((face_center_x - img_center_x) / w) * 100
    offset_y = ((face_center_y - img_center_y) / h) * 100

    horizontal_direction = "right" if offset_x > 0 else "left"
    vertical_direction = "down" if offset_y > 0 else "up"

    direction_1 = (horizontal_direction, str(round(abs(offset_x), 2)) + "%", offset_x)
    direction_2 = (vertical_direction, str(round(abs(offset_y), 2)) + "%", offset_y)

    return direction_1, direction_2


# Определение поворота головы, наклона и овала лица
def get_head_rotation_incline_and_face_oval(image):
    blur_image = cv2.blur(image, (5, 5))
    with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(blur_image)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Получаем координаты ключевых точек
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose = landmarks[1]
            chin = landmarks[152]  # Подбородок
            forehead = landmarks[10]  # Точка на лбу (верхняя часть)

            # Вычисляем расстояние между долями
            eye_distance = np.sqrt((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2)
            nose_to_left_eye = np.sqrt((left_eye.x - nose.x) ** 2 + (left_eye.y - nose.y) ** 2)
            nose_to_right_eye = np.sqrt((right_eye.x - nose.x) ** 2 + (right_eye.y - nose.y) ** 2)

            # Оценка угла поворота
            if nose_to_left_eye < 0.8 * eye_distance and nose_to_right_eye < 0.8 * eye_distance:
                head_rotation = "Straight"
            elif nose_to_left_eye > 1.1 * eye_distance:
                head_rotation = "Profile Left"
            elif nose_to_right_eye > 1.1 * eye_distance:
                head_rotation = "Profile Right"
            else:
                head_rotation = "Three Quarter"

            # Возвращаем информацию о наклоне головы
            if (chin.x < nose.x + 0.05 * eye_distance) and (nose.x - 0.05 * eye_distance < chin.x):  # Прямо
                head_incline = "Flat"
            elif chin.x >= nose.x + 0.05 * eye_distance:  # Влево
                head_incline = "Tilted left"
            else:
                head_incline = "Tilted right"

            return head_rotation, head_incline, (int(chin.x * image.shape[1]), int(chin.y * image.shape[0])), \
                (int(nose.x * image.shape[1]), int(nose.y * image.shape[0])), \
                (int(left_eye.x * image.shape[1]), int(left_eye.y * image.shape[0])), \
                (int(right_eye.x * image.shape[1]), int(right_eye.y * image.shape[0])), \
                (int(forehead.x * image.shape[1]), int(forehead.y * image.shape[0]))

    return None, None, None, None, None, None, None


# Детекция лиц: проверка красных глаз
def check_red_eyes(image):
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))
    return "Red Eyes Detected" if len(eyes) <= 0 else "No Red Eyes"

def detect_face_shadow_level(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return (None, None), None

    x, y, w, h = faces[0]
    face_region = gray[y:y+h, x:x+w]

    shadow_threshold = 50
    shadow_ratio_threshold = 0.09

    shadow_mask = face_region < shadow_threshold
    shadow_ratio = np.sum(shadow_mask) / shadow_mask.size

    shadow_level = "High" if shadow_ratio > shadow_ratio_threshold else "Medium"
    shadow_percent = round(shadow_ratio * 100, 2)

    return (shadow_level, str(round(abs(shadow_percent), 2))+"%"), shadow_percent

# Основная функция анализа изображения
def analyze_image(image_path):
    # Загружаем изображение
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # Размер изображения

    # Сегментация изображения
    bbox = segment_image(image)
    if bbox is None:
        return None, None, None, None  # Возвращаем None, если лицо не найдено

    # Получаем результаты
    background_status = check_blur(image)
    background_color = check_background_color(image)
    noise_level = classify_noise(image)
    contrast_level = classify_contrast(image)
    brightness = calculate_brightness(image)
    illumination_status = check_illumination(image)
    red_eye_status = check_red_eyes(image)
    shadow_level, shadow_level_i = detect_face_shadow_level(image)

    # Анализируем размер
    size_category, face_size_percentage = analyze_size(bbox, image.shape)
    direction_1, direction_2 = calculate_face_center_offset(image)
    # Определяем поворот головы, наклон и овал лица
    head_rotation, head_incline, chin_coords, nose_coords, left_eye_coords, right_eye_coords, forehead_coords = \
        get_head_rotation_incline_and_face_oval(image)

    direction_1_s = direction_1[:len(direction_1) - 1]
    direction_2_s = direction_2[:len(direction_2) - 1]

    # Возвращаем отчет и обработанное изображение
    report = {
        "Image Size (Pixels)": f"{w} x {h}",
        "Bounding Box": bbox,
        "Face Size (%)": face_size_percentage,
        "Size Category": size_category,
        "Deviation from the center (horizontal)": direction_1_s,
        "Deviation from the center (vertical)": direction_2_s,
        "Background Status": background_status,
        "Background Color": background_color,
        "Noise Level": noise_level,
        "Contrast Level": contrast_level,
        "Brightness (%)": brightness,
        "Illumination Status": illumination_status,
        "Red Eye Status": red_eye_status,
        "Face Shadows Level": shadow_level,
        "Head Rotation": head_rotation,
        "Head Incline": head_incline
    }

    # Создаем изображение с выделением лица
    annotated_image = image.copy()
    cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0),
                  2)  # Зеленая рамка для лица

    # Рисуем линии наклона головы
    cv2.line(annotated_image, chin_coords, nose_coords, (255, 0, 255), 2)  # Фиолетовая линия наклона
    cv2.line(annotated_image, left_eye_coords, right_eye_coords, (0, 165, 255), 2)  # Оранжевая линия между зрачками

    # Обозначить верхнюю часть лба
    cv2.circle(annotated_image, forehead_coords, 5, (0, 0, 255), -1)  # Красный круг в области лба

    blur_image = cv2.blur(annotated_image, (5, 5))
    with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB))
        for face in results.multi_face_landmarks:
            for landmark in face.landmark:
                x = landmark.x
                y = landmark.y

                shape = image.shape
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])

                cv2.circle(annotated_image, (relative_x, relative_y), radius=1, color=(225, 0, 100), thickness=1)

    return report, annotated_image, direction_1[2], direction_2[2], shadow_level_i


# изменила сравнительный репорт
# Основная функция сравнения изображений
def compare_images(template_path, test_path):
    template_report, template_image, offset_template_x, offset_template_y, template_shadow_level = analyze_image(template_path)
    test_report, test_image, offset_test_x, offset_test_y, test_shadow_level = analyze_image(test_path)

    # Создание аннотированного изображения для тестового изображения
    if test_report and "Bounding Box" in test_report:
        test_bbox = test_report["Bounding Box"]
        cv2.rectangle(test_image, (test_bbox[0], test_bbox[1]),
                      (test_bbox[0] + test_bbox[2], test_bbox[1] + test_bbox[3]), (255, 0, 0),
                      2)  # Красная рамка для тестового изображения

    # Подготовка сравнительного отчета
    # Сравнение размера изображений
    s1 = template_image.size
    s2 = test_image.size
    if template_report["Image Size (Pixels)"] == test_report["Image Size (Pixels)"]:
        Image_Size = "Is The Same"
    elif s1 > s2:
        Image_Size = "Smaller"
    else:
        Image_Size = "Bigger"

    # Сравнение размера лица
    if template_report["Face Size (%)"] == test_report["Face Size (%)"]:
        Face_Size = "Is The Same"
    elif template_report["Face Size (%)"] > test_report["Face Size (%)"]:
        Face_Size = "Smaller"
    else:
        Face_Size = "Bigger"

    # Сравнение расположения лица
    # По горизонтали
    # if template_report["Deviation from the center (horizontal)"] == test_report["Deviation from the center (horizontal)"]:
    if (offset_template_x * offset_test_x >= 0) and (abs(offset_template_x - offset_test_x) < 3) :
        Deviation_H = "Is The Same"
    elif test_report["Deviation from the center (horizontal)"][0] == "right":
        Deviation_H = "The Face Is To The Right"
    else:
        Deviation_H = "The Face Is To The Left"

    # По вертикале
    # if template_report["Deviation from the center (vertical)"] == test_report["Deviation from the center (vertical)"]:
    if (offset_template_y * offset_test_y >= 0) and (abs(offset_template_y - offset_test_y) < 3) :
        Deviation_V = "Is The Same"
    elif test_report["Deviation from the center (vertical)"][0] == "up":
        Deviation_V = "The Face Is Higher"
    else:
        Deviation_V = "The Face Is Below"

    # Размытие фона
    if template_report["Background Status"] == test_report["Background Status"]:
        Background_Status = "Is The Same"
    elif test_report["Background Status"] == "Blurred":
        Background_Status = "is more blurred"
    else:
        Background_Status = "is clearer"

    # Сравнение цвета фона
    if template_report["Background Color"] == test_report["Background Color"]:
        Background_Color = "Is The Same"
    else:
        Background_Color = "Is Not The Same"

    # Сравнение уровня шума
    if template_report["Noise Level"] == test_report["Noise Level"]:
        noise_level = "Is The Same"
    elif template_report["Noise Level"] == "High":
        noise_level = "Is Below"
    elif template_report["Noise Level"] == "Low":
        noise_level = "Is Higher"
    else:
        if test_report["Noise Level"] == "High":
            noise_level = "Is Higher"
        else:
            noise_level = "Is Below"

    # сравнение контраста
    if template_report["Contrast Level"] == test_report["Contrast Level"]:
        contrast_level = "Is The Same"
    elif template_report["Contrast Level"] == "High":
        contrast_level = "Is Below"
    elif template_report["Contrast Level"] == "Low":
        contrast_level = "Is Higher"
    else:
        if test_report["Contrast Level"] == "High":
            contrast_level = "Is Higher"
        else:
            contrast_level = "Is Below"

    # уровень яркости
    if template_report["Brightness (%)"] == test_report["Brightness (%)"]:
        brightness = "Is The Same"
    elif template_report["Brightness (%)"] > test_report["Brightness (%)"]:
        brightness = "Is Below"
    else:
        brightness = "Is Higher"

    # уровень отсвещенности
    if template_report["Illumination Status"] == test_report["Illumination Status"]:
        illumination_status = "Is The Same"
    elif template_report["Illumination Status"] == "Even Light":
        illumination_status = "Is Below"
    else:
        illumination_status = "Is Higher"

    # красные глаза
    if template_report["Red Eye Status"] == test_report["Red Eye Status"]:
        red_eye_status = "Is The Same"
    else:
        red_eye_status = "Is Not The Same"

    # уровень теней
    if template_report["Face Shadows Level"][0] == test_report["Face Shadows Level"][0]:
        if (template_shadow_level * test_shadow_level >= 0) and (abs(template_shadow_level - test_shadow_level) < 3):
            shadow_level = "Is The Same"
        elif template_shadow_level > test_shadow_level:
            shadow_level = "Is Below"
        else:
            shadow_level = "Is Higher"
    elif test_report["Face Shadows Level"][0] == "High":
        shadow_level = "Is Higher"
    else:
        shadow_level = "Is Below"
    
    # Поворот головы  'Straight', 'Profile Left', 'Profile Right', 'Three Quarter'
    if template_report["Head Rotation"] == test_report["Head Rotation"]:
        head_rotation = "Is The Same"
    elif template_report["Head Rotation"] == "Profile Left":

        head_rotation = "To The Right"
    elif template_report["Head Rotation"] == "Profile Right":
        head_rotation = "To The Left"
    elif template_report["Head Rotation"] == "Straight":
        if test_report["Head Rotation"] == "Profile Left" or test_report["Head Rotation"] == "Profile Right":
            head_rotation = "Is Not Straight: Profile"
        else:
            head_rotation = "Is Not Straight: Three Quarter"
    else:
        head_rotation = "Is Not Three Quarter"

    # Наклон головы  'Flat', 'Tilted left', 'Tilted right'
    if template_report["Head Incline"] == test_report["Head Incline"]:
        head_incline = "Is The Same"
    elif template_report["Head Incline"] == "Tilted left":
        head_incline = "Tilted To The Right"
    elif template_report["Head Incline"] == "Tilted right":
        head_incline = "Tilted To The Left"
    else:
        if test_report["Head Incline"] == "Tilted left":
            head_incline = "Tilted To The Left"
        else:
            head_incline = "Tilted To The Right"

    comparison_report = {
        # "Template Report": template_report,
        # "Test Report": test_report,
        "Comparison": {
            "Image Size (Pixels)": Image_Size,
            "Face Size (%)": Face_Size,
            "Deviation from the center (horizontal)": Deviation_H,
            "Deviation from the center (vertical)": Deviation_V,
            "Background Status": Background_Status,
            "Background Color": Background_Color,
            "Noise Level": noise_level,
            "Contrast Level": contrast_level,
            "Brightness (%)": brightness,
            "Illumination Status": illumination_status,
            "Red Eye Status": red_eye_status,
            "Face Shadows Level": shadow_level,
            "Head Rotation": head_rotation,
            "Head Incline": head_incline
        }
    }

    return json.dumps(template_report), json.dumps(test_report), json.dumps(
        comparison_report), template_image, test_image


# Интерфейс Gradio
iface = gr.Interface(
    fn=compare_images,
    inputs=[
        gr.Image(type="filepath", label="Template Image"),
        gr.Image(type="filepath", label="Test Image")
    ],
    outputs=[
        gr.JSON(label="Template Analysis Report"),
        gr.JSON(label="Test Analysis Report"),
        gr.JSON(label="Comparative Analysis Report"),
        gr.Image(label="Annotated Template Image"),
        gr.Image(label="Annotated Test Image")
    ],
    title="Image Comparison Tool",
    description="Upload a template image and a test image to analyze their similarity."
)

# Запуск интерфейса
if __name__ == "__main__":
    iface.launch()
# share=True
