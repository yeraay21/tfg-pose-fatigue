from matplotlib.pyplot import show

from ultralytics import YOLO

model = YOLO("yolo26m-pose.pt")   # modelo ya entrenado para detección de poses

results = model.predict("dataset/bicep_curl/correct/subject_001_bicep_curl_good_side.mp4", save=True, show=True)       # ejecuta la detección


# results[0].save("test_pose.jpg")  # guarda la imagen con el esqueleto

# print(results[0].keypoints.xy.shape)
# print(results[0].keypoints.conf.shape)
# print(results[0].keypoints.xy)

for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)