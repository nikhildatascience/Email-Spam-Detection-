
# Email Spam Detection Using Machine Learning  

## Overview  
This project utilizes Machine Learning (Naïve Bayes) to classify emails as Spam or Ham (Not Spam).  

### Features:  
- Real-time classification using a trained ML model  
- Spam keyword highlighting for better understanding  
- Performance evaluation with Accuracy, Precision, Recall, and F1-score  
- Interactive web interface built with Gradio  
- Retraining capability to improve model accuracy  

---

## Project Structure  

```
Email-Spam-Detection  
│── Datasets.xlsb         → Dataset used for training the model  
│── LICENSE               → Open-source MIT license  
│── README.md             → Project documentation  
│── app.py                → Main application script  
│── metadata.yaml         → Configuration for Hugging Face Spaces  
│── requirements.txt      → Dependencies required for the project  
│── spam_model.pkl        → Trained Naïve Bayes model  
│── spam_vectorizer.pkl   → CountVectorizer for text processing  
```

---

## How to Run Locally  

1. Clone this repository:  
```bash
git clone https://github.com/nikhildatascience/Email-Spam-Detection.git
cd Email-Spam-Detection
```
2. Install dependencies:  
```bash
pip install -r requirements.txt
```
3. Run the application:  
```bash
python app.py
```
4. Open in a browser:  
Once the app starts, a Gradio link will be generated. Open it to use the spam detector.  

---

## Model Performance  

This model is trained using Naïve Bayes with TF-IDF vectorization and evaluated using the following metrics:  

| Metric     | Score  |
|------------|--------|
| Accuracy   | 99.52%  |
| Precision  | 100.00% |
| Recall     | 99.01%  |
| F1-Score   | 99.50%  |

---

## Retraining the Model  
To improve accuracy, users can add new emails and retrain the model:  
1. Enter the email content  
2. Select "Spam" or "Ham"  
3. Click "Save & Retrain" to update the model  

---

## Live Demo  
Try the live demo on Hugging Face:  
[Live Demo](https://huggingface.co/spaces/ABHI010/NIKHIL14)  

---

## License  
This project is open-source under the MIT License. You are free to modify and use it.  
