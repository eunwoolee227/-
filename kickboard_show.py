from ultralytics import YOLO
import cv2

image_path = "ex3.jpg"
model_1 = YOLO("yolo11s.pt")
model_2 = YOLO("kb_yolo_ew.pt")

results_1 = model_1(image_path)
results_2 = model_2(image_path)
results_3 = model_2(image_path, conf=0.8)

cv2.imshow("results", results_3[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()

results_1[0].save("kickboard_result1.jpg")
results_2[0].save("kickboard_result2.jpg")
results_3[0].save("kickboard_result3.jpg")