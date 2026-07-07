import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)

print("Press 'Q' to Quit")

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to access webcam.")
        break

   
    results = model.track(frame, persist=True)

    
    annotated_frame = results[0].plot()

    
    cv2.imshow("CodeAlpha Object Detection and Tracking", annotated_frame)

   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()