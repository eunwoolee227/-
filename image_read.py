from ultralytics import solutions
import cv2


image_path = "ex2.jpg"
img = cv2.imread(image_path)


counter = solutions.ObjectCounter(
  show=False,        
  model="yolo11s.pt",
  classes = [2],
  show_conf = False,
  show_labels = False
)


results = counter(img)


cv2.putText(results.plot_im, f"car number : {results.total_tracks}", 
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


blurrer = solutions.ObjectBlurrer(
  show =False,       
  model="yolo11s.pt",
  classes = [0],
  show_conf = False,
  show_labels = False,
)


results2 = blurrer(results.plot_im)


cv2.imshow("results", results2.plot_im)

cv2.waitKey(0)    
cv2.destroyAllWindows()

cv.imwrite("ex2_results2.py")