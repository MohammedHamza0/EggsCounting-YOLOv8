import cv2
import os
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter

os.chdir(r"F:\YOLO Projects\EggsCount")

model = YOLO("EggsBest.pt")


# Define line points
line_points = [[256, 40],[252, 606]]

counter = ObjectCounter(
    view_img=False,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

cap = cv2.VideoCapture("egg.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("number of the frames have finished")
        break
    else:
        frame = cv2.resize(frame, (700, 700))
        
        tracks = model.track(frame, persist=True, show=False, conf=0.25)
        frame = counter.start_counting(frame, tracks)
        
        cv2.imshow("EggsCounter", frame)
        if cv2.waitKey(1) == 27:
            break
        
        
cap.release()
cv2.destroyAllWindows()