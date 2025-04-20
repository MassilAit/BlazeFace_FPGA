import cv2
from camera_util import start_camera, read_image, show_frame
from serial_com import *
from model_util import preprocess, postprocess, load_anchors

anchors= load_anchors("anchors.npy")  

cap = start_camera()
ser = intit_serial_com()

while cap.isOpened():

    frame, resized_frame = read_image(cap)

    input = preprocess(resized_frame)

    send_frame(ser, input)

    output = read_output(ser)

    detections = postprocess(output, anchors)

    show_frame(frame, detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
cap.release()
cv2.destroyAllWindows()