import cv2
import numpy as np
import pytesseract
import pyttsx3
import threading
import queue

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Speech queue to handle speech requests
speech_queue = queue.Queue()

def tts_worker():
    while True:
        text = speech_queue.get()
        if text is None:  # Exit signal
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def speak(text):
    # Clear the queue if it's getting too long to avoid delays in reading new detections
    if speech_queue.qsize() > 5:
        with speech_queue.mutex:
            speech_queue.queue.clear()

    # Put the new speech task into the queue
    speech_queue.put(text)

# Improved distance calculation
def calculate_distance(box, frame_width, known_width=0.5, focal_length=700):
    """Estimate the distance of an object using its bounding box width."""
    return (known_width * focal_length) / (box[2])

def process_frame(frame, detected_objects_cache):
    height, width, _ = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Loop through each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Confidence threshold for detection
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression for overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)  # Lowered thresholds

    new_detections = set()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

            # Calculate distance (example, using box width for approximation)
            distance = calculate_distance(boxes[i], width)
            detection_info = f"{label} at {round(distance, 2)} meters"

            # Add new detections
            if detection_info not in detected_objects_cache:
                new_detections.add(detection_info)

    # Immediately speak out new detections
    if new_detections:
        detected_objects_cache.update(new_detections)
        speak(f"Detected: {', '.join(new_detections)}")

    # OCR to detect text in the frame
    text = pytesseract.image_to_string(frame)
    if text.strip():
        if text not in detected_objects_cache:  # Prevent repetition
            cv2.putText(frame, "Detected Text", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            speak(f"Detected text: {text.strip()}")
            detected_objects_cache.add(text.strip())

    return frame

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Cache for detected objects to avoid repetition
    detected_objects_cache = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame (detect objects and text)
        processed_frame = process_frame(frame, detected_objects_cache)

        # Display the frame
        cv2.imshow('Blind Assistance System', processed_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Signal the TTS thread to exit
    speech_queue.put(None)
    tts_thread.join()

if __name__ == "__main__":
    main()
