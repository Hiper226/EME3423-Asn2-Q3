import cv2
import numpy as np

confThreshold = 0.8
count_apple = 0 # each 1 dollar
count_orange = 0 # each 3 dollar
count_banana = 0 # each 5 dollar

image_path = 'fruits.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from '{image_path}'")
    print("Make sure the image file exists in the same folder as this script.")
    exit()

print(f"Image loaded successfully: {image_path}")


classesFile = 'coco80.names'
classes = []
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()

print(f"Loaded {len(classes)} classes")

# Load the configuration and weights file
net = cv2.dnn.readNetFromDarknet('yolov3-608.cfg', 'yolov3-608.weights')

# Use OpenCV as backend and use CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get image dimensions
height, width, ch = img.shape

# Prepare blob
blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layerNames = net.getLayerNames()
output_layers_names = net.getUnconnectedOutLayersNames()

# Forward pass
LayerOutputs = net.forward(output_layers_names)

bboxes = []
confidences = []
class_ids = []

for output in LayerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confThreshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            bboxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(bboxes), 3))

# Draw detections
if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = bboxes[i]
        label = str(classes[class_ids[i]])
        if label == "apple":
            count_apple+=1
        if label == "orange":
            count_orange+=1
        if label == "banana":
            count_banana+=1
        confidence = str(100*round(confidences[i], 2)) + "%"
        color = colors[i]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 20),
                    font, 2, (0, 0, 0), 2)

# put count text
cv2.putText(img, "apple = " + str(count_apple), (30,30),
            font, 2, (0, 0, 0), 2)
cv2.putText(img, "orange = " + str(count_orange), (30,60),
            font, 2, (0, 0, 0), 2)
cv2.putText(img, "banana = " + str(count_banana), (30,90),
            font, 2, (0, 0, 0), 2)
total_cost = count_apple*1 + count_orange*3 + count_banana*5
cv2.putText(img, "Price = " + str(total_cost) + "$", (30,120),
            font, 2, (0, 0, 0), 2)
# Show the result
cv2.imshow('YOLO Object Detection - Image', img)

print("Press any key to close the window...")
cv2.waitKey(0)          # Wait until any key is pressed
cv2.destroyAllWindows()