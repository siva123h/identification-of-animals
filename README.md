# identification-of-animals
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Load the pre-trained SSD MobileNetV2 model from TF-Hub
# ---------------------------------------------------------
print("Loading model from TensorFlow Hub...")
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded successfully!")


# ---------------------------------------------------------
# FUNCTION: Run object detection on a single image
# ---------------------------------------------------------
def detect_objects(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f" animals load image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (320, 320))
    input_tensor = np.expand_dims(img_resized, axis=0)

    # Run detection
    result = model(input_tensor)

    boxes = result["detection_boxes"][0].numpy()
    class_ids = result["detection_classes"][0].numpy().astype(int)
    scores = result["detection_scores"][0].numpy()

    print(f"\n Results for: {image_path}")
    print("------------------------------------")
    for i in range(len(scores)):
        if scores[i] > 0.5:
            print(f"Object {i+1}: class={class_ids[i]}, score={scores[i]:.2f}")

    # Show image
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------
# FUNCTION: Detect objects in all images inside a folder
# ---------------------------------------------------------
def detect_objects_in_folder(path):
    if os.path.isfile(path):
        print("animals image detected → running single image detection.")
        detect_objects(path)
        return

    if not os.path.isdir(path):
        print("Error: Path is not a folder or file.")
        return

    image_files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(image_files) == 0:
        print(" images found in folder!")
        return

    for img_file in image_files:
        full_path = os.path.join(path, img_file)
        detect_objects(full_path)


# ---------------------------------------------------------
# ASK USER FOR INPUT
# ---------------------------------------------------------
user_input = input("Enter image path OR folder path: ")

detect_objects_in_folder(user_input)
