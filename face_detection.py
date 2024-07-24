import onnxruntime
import cv2
import numpy as np

# COCO class names
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", 
               "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
               "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", 
               "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
               "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", 
               "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
               "scissors", "teddy bear", "hair drier", "toothbrush"]

# Load YOLOv5 ONNX model
onnx_model_path = 'yolov5s.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)

# Preprocess the input image
def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file {image_path} not found.")
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0  # Normalize to [0, 1]
    return img

# Run inference on the preprocessed image
def run_inference(image):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image[None, :, :, :]})
    return outputs

# Post-process the output
def postprocess(outputs, conf_threshold=0.25, iou_threshold=0.45):
    boxes, scores, class_ids = [], [], []
    output = outputs[0][0]
    for detection in output:
        if detection[4] > conf_threshold:  # Confidence threshold
            box = detection[:4]
            score = detection[4]
            class_id = np.argmax(detection[5:])
            if detection[5 + class_id] > iou_threshold:  # IoU threshold
                boxes.append(box)
                scores.append(score)
                class_ids.append(class_id)
    return boxes, scores, class_ids

# Display results
def display_results(image_path, boxes, scores, class_ids):
    img = cv2.imread(image_path)
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main code execution
if __name__ == '__main__':
    image_path = 'images/image8.jpg'  
    image = preprocess(image_path)
    outputs = run_inference(image)
    boxes, scores, class_ids = postprocess(outputs)
    print("Detected objects:")
    for class_id, score in zip(class_ids, scores):
        print(f"Class: {class_names[class_id]}, Score: {score:.2f}")
    display_results(image_path, boxes, scores, class_ids)
