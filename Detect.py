import cv2
import pytesseract
from picamera2 import Picamera2


def comprobar_pattern(text):
    if "K" in text:
        index = text.find("K")
        if text[index + 1] == "I" and text[index + 2] == "W" and text[index + 3] == "I":
            return True

    return False


if "__main__" == __name__:
    # Modelo entrenado de ML para la detección de las matrículas.
    harcascade = "haarcascade_russian_plate_number.xml"

    # TRACKER
    tracker = cv2.TrackerCSRT_create()
    inicializado = False

    # Captura de video
    picam = Picamera2()
    picam.preview_configuration.main.size = (3460, 1440)
    picam.preview_configuration.main.format = "RGB888"
    picam.camera_usb_options = "-r 2592x1944 -f10"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # Minima área para ver potential plate
    min_area = 30000

    # pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

    # Frame
    frame = picam.capture_array()
    BB = cv2.selectROI(frame, True)
    # Cerrar la ventana después de la selección
    cv2.destroyWindow("ROI Selection")

    while True:
        # Frame
        frame = picam.capture_array()

        if inicializado:
            # Trackea
            track_success, BB = tracker.update(frame)
            if track_success:
                top_left = (int(BB[0]), int(BB[1]))
                bottom_right = (int(BB[0] + BB[2]), int(BB[1] + BB[3]))
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 5)

        # Convertir a gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Clasificador en cascada
        # Creando objecto para encontrar matriculas
        plate_detector = cv2.CascadeClassifier(harcascade)

        # Posibles matriculas
        plates = plate_detector.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in plates:
            # Area de la zon acon la matrícula
            area = w * h

            if area > min_area:
                # Pinta rectangulo del área
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Imagen ampliada a la zona
                plate_img = frame[y : y + h, x : x + w]
                # La guardamos
                # cv2.imwrite("plate_image_video.jpg", plate_img)

                # Pasamos la imagen de la matricula a binaria
                gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                gray_plate_img = cv2.GaussianBlur(gray_plate_img, (5, 5), 0)
                _, plate_img_binary = cv2.threshold(
                    gray_plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                # Hacemos opening de la imagen para eliminar objectos insignificantes
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                plate_img_binary = cv2.morphologyEx(
                    plate_img_binary, cv2.MORPH_OPEN, kernel
                )
                # Guardar
                # cv2.imwrite("binary_image_video.jpg", plate_img_binary)

                # Extraer texto
                plate_text = pytesseract.image_to_string(
                    plate_img_binary, config="--psm 7"
                )
                # print("text: ", plate_text)

                # Añadir texto
                cv2.putText(
                    frame,
                    "Extracted Text: " + plate_text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

                if comprobar_pattern(plate_text):
                    cv2.imwrite("plate_image_video.jpg", plate_img)
                    cv2.imwrite("binary_image_video.jpg", plate_img_binary)

                    # Inicializamos traker en el priner frame
                    if not inicializado:
                        tracker.init(frame, BB)
                        inicializado = True

        # Printeamos el frame con todas las placas detectadas
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
