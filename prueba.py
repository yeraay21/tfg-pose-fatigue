from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")   # modelo ya entrenado para detección de poses
results = model("test.jpg")       # ejecuta la detección
results[0].save("test_pose.jpg")  # guarda la imagen con el esqueleto

print(results[0].keypoints.xy.shape)
print(results[0].keypoints.conf.shape)
print(results[0].keypoints.xy)