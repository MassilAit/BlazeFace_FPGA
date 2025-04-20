import cv2

def start_camera():
    # Open webcam 
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit()


    # after you open the camera, but before your while‐loop:
    cv2.namedWindow("Live Face Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Live Face Detection",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

    return cap


def read_image(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Error: Failed to read frame from camera.")

    input_size = 128 
    resized_frame = cv2.resize(frame, (input_size, input_size))

    return frame, resized_frame


def draw_detections(frame, detections):
    h, w, _ = frame.shape
    for detection in detections:
        ymin, xmin, ymax, xmax = detection[:4]
        xmin, xmax = int(xmin * w), int(xmax * w)
        ymin, ymax = int(ymin * h), int(ymax * h)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


def show_frame(frame, detections):
    if len(detections) > 0:
        draw_detections(frame, detections)

    cv2.resize(frame, (128, 128))
    # Display the frame with detections
    cv2.imshow("Live Face Detection", frame)

