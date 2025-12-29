import torch
import cv2

# Load the YOLOv5 model (small and fast)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally (change the value for vertical flip if needed)
    frame = cv2.flip(frame, 1)  # 1 = Horizontal flip, 0 = Vertical flip, -1 = Both

    # Convert the image to RGB (YOLO expects RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img)

    # Convert the results to pandas dataframe
    results_df = results.pandas().xywh[0]  # Access the first image in results

    # Filter to only show "person" class (class ID for "person" is 0 in COCO dataset)
    results_df = results_df[results_df['class'] == 0]  # Filter for 'person' class

    # Render results (Bounding boxes)
    results.render()  # This will draw boxes on the image

    output = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

    # Show the frame with bounding boxes
    cv2.imshow("YOLOv5 People Detection (Flipped)", output)

    # Snapshot on 's' key press
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("snapshot.jpg", output)  # Save the snapshot
        print("Snapshot saved as 'snapshot.jpg'")

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()