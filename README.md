# PoseDetection - ASL Translator  

A machine learning project that utilizes **TensorFlow HandPose** and **K-Nearest Neighbors (KNN)** to recognize and predict **American Sign Language (ASL)** gestures. The model is trained to achieve the highest possible accuracy and then used to translate ASL into actual sentences.  

---

## How It Works  

1. **Training the Model**  
   - The model is trained using **TensorFlow HandPose** to recognize different ASL hand gestures.  
   - It is continuously corrected and refined to improve accuracy.  
   - Once the model reaches high accuracy, it is saved and downloaded for future use.  

2. **Predicting ASL in Real-Time**  
   - The trained model is loaded in `predict.js`.  
   - It captures hand gestures using a webcam and translates them into ASL letters.  
   - The system **only adds a letter to the sentence when it is 100% sure** of the prediction, ensuring high accuracy.  

---

## How to Run the Project  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/posedetection.git
   ```

2. **Run the ASL Translator**  
   - Open `predict.html` in your browser to test the live ASL recognition.  
   - Alternatively, you can run `index.html` for a different interface.  

---

## Features  

✅ Uses **TensorFlow HandPose** to detect hand landmarks.  
✅ Employs **KNN (K-Nearest Neighbors)** for ASL gesture classification.  
✅ Ensures **high accuracy** before adding a letter to a sentence.  
✅ Translates ASL gestures into readable text in real-time.  

---

This project aims to bridge the gap between ASL users and non-signers by providing an **accurate and efficient ASL translation tool**. 🚀  

Let me know if you need any modifications! 😊
