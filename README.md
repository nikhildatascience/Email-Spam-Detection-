# 📧 Email Spam Detection Using Machine Learning  

![Spam Detection](https://cdn.pixabay.com/photo/2016/11/19/15/26/email-1839873_1280.jpg)  

## 🚀 Overview  
This project uses Machine Learning (Naïve Bayes) to classify emails as Spam or Ham (not spam). It provides:  
- Real-time email classification using a trained ML model.  
- Spam keyword highlighting for better understanding.  
- Performance evaluation with Accuracy, Precision, Recall, and F1-score.  
- Interactive web interface built with Gradio.  
- Retraining capability to improve model accuracy.  

## 🛠️ Features  
✅ Spam Detection – Classifies emails as Spam or Ham.  
✅ Confidence Score – Shows how sure the model is.  
✅ Keyword Highlighting – Marks common spam words.  
✅ Performance Metrics – Displays Accuracy, Precision, Recall, and F1-score.  
✅ Retraining – Allows adding new emails for continuous learning.  

## 📂 Files in this Repository  

📁 Email-Spam-Detection  
│── 📜 app.py               # Main application script  
│── 📜 spam_model.pkl       # Trained Naïve Bayes model  
│── 📜 spam_vectorizer.pkl  # CountVectorizer for text processing  
│── 📜 requirements.txt     # Dependencies needed to run the project  
│── 📜 metadata.yaml        # Configuration for Hugging Face Spaces  
│── 📜 LICENSE              # Open-source MIT license  
│── 📜 README.md            # Project documentation  

## 🖥️ How to Run Locally  
1️⃣ Clone this repository  
```bash
git clone https://github.com/nikhildatascience/Email-Spam-Detection.git
cd Email-Spam-Detection
```
2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```
3️⃣ Run the app  
```bash
python app.py
```
4️⃣ Open in browser  
After running, a Gradio link will be generated. Open it to use the spam detector!  

## 📊 Model Performance  
The model has been trained using Naïve Bayes with TF-IDF vectorization and evaluated using:  
- Accuracy: ~95%  
- Precision: High spam detection  
- Recall: Catches most spam emails  
- F1-Score: Balanced measure  

## 🎯 Retraining the Model  
Want to improve accuracy? Add new emails and retrain:  
1. Enter the email content.  
2. Select "Spam" or "Ham".  
3. Click "Save & Retrain" to update the model.  

## 🌍 Live Demo  
You can access the live demo of this project on Hugging Face:  
🔗 [Live Demo](https://huggingface.co/spaces/ABHI010/NIKHIL14)  

## 📜 License  
This project is open-source under the MIT License. Feel free to modify and use it!  

---  
💡 Created by [Nikhil N C](https://github.com/nikhildatascience) 🚀  
```
