Face Detection with YOLOv5 and ONNX
---------------------------------------------------------------------------------------------------------------------------

*Overview*

This project demonstrates the implementation of a face detection system using the YOLOv5 model converted to ONNX format. The system processes input images to accurately detect and localize faces, drawing bounding boxes around detected faces and providing confidence scores.


Installation
---------------------------------------------------------------------------------------------------------------------------

1. Clone the Repository:

   git clone https://github.com/shettydevika/Face--Detection.git
   
   cd face-detection-onnx

3. Create and Activate a Virtual Environment:

   python -m venv venv
   
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

5. Install Dependencies:

     pip install -r requirements.txt

Usage
---------------------------------------------------------------------------------------------------------------------------

1. Prepare the Input Image:

  Place the image you want to test in the images directory.

2. Run the Face Detection Script:

    python face_detection.py

3. View Results:

The script will display the input image with detected faces marked by bounding boxes and labeled with confidence scores.

Results
---------------------------------------------------------------------------------------------------------------------------

The YOLOv5 ONNX model successfully detects faces in input images with high accuracy and efficiency. Detected faces are marked with bounding boxes, and confidence scores are displayed, indicating the reliability of the detections.

Screenshots
---------------------------------------------------------------------------------------------------------------------------

Figure 1: Test Image
![Alt Text](images/image8.jpg)

Figure 2: Face Detection Result
![Alt Text](result.png)

Figure 3: VS Code Result
![Alt Text](vs1.png)
![Alt Text](vs2.png)

Conclusion
---------------------------------------------------------------------------------------------------------------------------

The face detection project using the YOLOv5 ONNX model showcases the model's effectiveness in identifying and localizing faces in real-time. This project highlights the practical application of advanced neural network models in various domains, providing valuable hands-on experience and fostering a deeper understanding of AI technologies.
