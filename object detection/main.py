import cv2
import numpy as np
import pyttsx3
import time


# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Handle layer names for YOLO

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak detection labels
def speak_detection(labels):
    # Combine labels into a single string
    if len(labels) > 1:
        sentence = "I can see a " + " and a ".join(labels)
    else:
        sentence = "I can see a " + labels[0]
    print(sentence)
    engine.say(sentence)
    engine.runAndWait()

# Open webcam
cap = cv2.VideoCapture(0)

# Set a lower resolution for the video stream (e.g., 640x480)
cap.set(3, 640)
cap.set(4, 480)

frame_count = 0  # Variable to track frames

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    # Skip every 3rd frame to reduce processing load
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Extract information from detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:  # Lowered confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove duplicates
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)  # Reduced NMS threshold
    detected_labels = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_labels.append(label)
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            

    # Speak all detected labels in one sentence
    if detected_labels:
        speak_detection(detected_labels)

    # Show frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
