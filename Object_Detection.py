import cv2
import numpy as np
import pyttsx3

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)

detected_objects = []

engine = pyttsx3.init()

focal_length = 1000  
known_object_height = 0.2 


def calculate_distance(object_height_pixels):
    distance = (known_object_height * focal_length) / object_height_pixels
    return distance

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    output_layer_names = net.getUnconnectedOutLayersNames()

    outputs = net.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        detected_objects.append({
            "label": label,
            "confidence": confidence,
            "height_pixels": h
        })

        distance = calculate_distance(h)

        object_info = f"Label: {label}, Confidence: {confidence:.2f}, Height: {h} pixels, Distance: {distance:.2f} meters"
        engine.say(object_info)
        engine.runAndWait()

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Height: {h} pixels, Distance: {distance:.2f} meters', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Object Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

for obj in detected_objects:
    print(f"Label: {obj['label']}, Confidence: {obj['confidence']:.2f}, Height: {obj['height_pixels']} pixels, Distance: {calculate_distance(obj['height_pixels']):.2f} meters")
