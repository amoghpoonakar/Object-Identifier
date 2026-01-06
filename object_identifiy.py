from ultralytics import YOLO
import cv2

DETECT_PEOPLE = False
FRAME_SKIP = 3

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

window_name = "YOLOv8 Object Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0
last_annotated = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    if frame_count % FRAME_SKIP == 0 or last_annotated is None:
        try:
            if DETECT_PEOPLE:
                results = model(
                    frame,
                    imgsz=640,
                    conf=0.45,
                    iou=0.45,
                    verbose=False
                )
            else:
                results = model(
                    frame,
                    imgsz=640,
                    conf=0.45,
                    iou=0.45,
                    classes=list(range(1, 80)),
                    verbose=False
                )

            last_annotated = results[0].plot()

        except Exception as e:
            print("Inference error:", e)
            break

    cv2.imshow(window_name, last_annotated)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("snapshot.jpg", last_annotated)

cap.release()
cv2.destroyAllWindows()
