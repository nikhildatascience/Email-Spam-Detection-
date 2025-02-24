import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import gradio as gr
import re

# Load and preprocess dataset
dataset = pd.read_csv('Email_spam_niki.csv', on_bad_lines='skip', engine='python')

# Drop rows where 'spam' or 'text' is NaN and convert 'spam' to numeric
dataset.dropna(subset=['spam', 'text'], inplace=True)
dataset['spam'] = pd.to_numeric(dataset['spam'], errors='coerce')

# Remove any rows where 'spam' is NaN after conversion and convert 'spam' to integers
dataset.dropna(subset=['spam'], inplace=True)
dataset['spam'] = dataset['spam'].astype(int)

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['text'])
y = dataset['spam']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'spam_vectorizer.pkl')

# Reload for consistency
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('spam_vectorizer.pkl')

# List of spammy keywords
spam_keywords = [
    "win", "free", "urgent", "money", "credit", "loan", "offer", "buy now",
    "limited time", "click here", "guaranteed", "congratulations", "winner"
]

# Helper function to highlight spammy keywords
def highlight_keywords(text):
    highlighted = text
    for keyword in spam_keywords:
        pattern = re.compile(rf"(\b{keyword}\b)", re.IGNORECASE)
        highlighted = pattern.sub(f"<span class='highlight'>{keyword}</span>", highlighted)
    return highlighted

# Prediction function
def classify_email(email_text):
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    confidence = model.predict_proba(email_vector).max() * 100
    result = "Spam" if prediction[0] == 1 else "Ham"

    highlighted_text = highlight_keywords(email_text)
    color = "red" if result == "Spam" else "green"
    emoji = "📧" if result == "Ham" else "⚠️"
    advice = "<b>Be careful!</b> This might be a scam." if result == "Spam" else "<b>This email seems safe.</b>"

    return {
        "result": f"<span style='color: {color}; font-size: 1.5em;'>{emoji} {result}</span>",
        "confidence": f"{confidence:.2f}%",
        "highlighted": highlighted_text,
        "spammy_keywords": ", ".join(
            [kw for kw in spam_keywords if kw.lower() in email_text.lower()]
        ),
        "advice": advice
    }

# Generate performance metrics
def generate_performance_metrics():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save plot as a base64 string
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    return {
        "accuracy": f"{accuracy:.2%}",
        "precision": f"{report['1']['precision']:.2%}",
        "recall": f"{report['1']['recall']:.2%}",
        "f1_score": f"{report['1']['f1-score']:.2%}",
        "confusion_matrix_plot": img_base64,
    }

# Function to add new email data and retrain the model
def save_and_retrain(email_text, label):
    try:
        # Convert label to numeric value (0 for Ham, 1 for Spam)
        label_numeric = 1 if label == "Spam" else 0
        
        # Add the new data to the dataset
        new_data = pd.DataFrame({"text": [email_text], "spam": [label_numeric]})
        global dataset, X, y, model, vectorizer
        dataset = pd.concat([dataset, new_data], ignore_index=True)

        # Vectorize the updated text data
        X = vectorizer.fit_transform(dataset['text'])
        y = dataset['spam']

        # Retrain the model
        model.fit(X, y)

        # Save the updated model and vectorizer
        joblib.dump(model, 'spam_model.pkl')
        joblib.dump(vectorizer, 'spam_vectorizer.pkl')

        return "Model retrained successfully with new data!"
    except Exception as e:
        return f"Error while retraining: {str(e)}"

# Updated CSS
custom_css = """
body {
    font-family: 'Arial', sans-serif;
    background-image: url('https://cdn.pixabay.com/photo/2016/11/19/15/26/email-1839873_1280.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #333;
}
h1, h2, h3 {
    text-align: center;
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
}
.gradio-container {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
}
button {
    background-color: #1e90ff;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1.2em;
    transition: transform 0.2s, background-color 0.3s;
}
button:hover {
    background-color: #1c86ee;
    transform: scale(1.05);
}
.highlight {
    background-color: #ffeb3b;
    font-weight: bold;
    padding: 0 3px;
    border-radius: 3px;
}
.metric {
    font-size: 1.2em;
    text-align: center;
    color: #ffffff;
    background-color: #4CAF50;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
}
"""

# Create Gradio Interface
def create_interface():
    performance_metrics = generate_performance_metrics()

    with gr.Blocks(css=custom_css) as interface:
        gr.Markdown("# 📩 Advanced Email Spam Classifier")
        gr.Markdown(
            """
            ### Enter the content of an email below to classify it as Spam or Ham.
            The tool uses **machine learning** to analyze email content, highlights spammy keywords, and shows key performance analytics.
            """
        )

        with gr.Row():
            with gr.Column():
                email_input = gr.Textbox(
                    lines=8, placeholder="Type or paste your email content here...", label="Email Content"
                )
            with gr.Column():
                result_output = gr.HTML(label="Classification Result")
                confidence_output = gr.Textbox(label="Confidence Score", interactive=False)
                highlighted_output = gr.HTML(label="Highlighted Text")
                keywords_output = gr.Textbox(label="Spam Keywords Detected", interactive=False)
                advice_output = gr.HTML(label="Advice")

        analyze_button = gr.Button("Analyze Email 🕵️‍♂️")

        def email_analysis_pipeline(email_text):
            results = classify_email(email_text)
            return (
                results["result"],
                results["confidence"],
                results["highlighted"],
                results["spammy_keywords"],
                results["advice"]
            )

        analyze_button.click(
            fn=email_analysis_pipeline,
            inputs=email_input,
            outputs=[result_output, confidence_output, highlighted_output, keywords_output, advice_output]
        )

        gr.Markdown("## 📊 Model Performance Analytics")
        with gr.Row():
            with gr.Column():
                gr.Textbox(value=performance_metrics["accuracy"], label="Accuracy", interactive=False, elem_classes=["metric"])
                gr.Textbox(value=performance_metrics["precision"], label="Precision", interactive=False, elem_classes=["metric"])
                gr.Textbox(value=performance_metrics["recall"], label="Recall", interactive=False, elem_classes=["metric"])
                gr.Textbox(value=performance_metrics["f1_score"], label="F1 Score", interactive=False, elem_classes=["metric"])
            with gr.Column():
                gr.Markdown("### Confusion Matrix")
                gr.HTML(f"<img src='data:image/png;base64,{performance_metrics['confusion_matrix_plot']}' style='max-width: 100%; height: auto;' />")

        gr.Markdown("## 🛠️ Save and Retrain the Model")
        with gr.Row():
            email_for_retraining = gr.Textbox(
                lines=8, placeholder="Enter the email content to label as Spam or Ham and retrain", label="Email Content"
            )
            label_input = gr.Radio(["Spam", "Ham"], label="Label", type="value")

        retrain_button = gr.Button("Save & Retrain Model")
        retrain_result = gr.Textbox(label="Retrain Result", interactive=False)

        retrain_button.click(
            fn=save_and_retrain,
            inputs=[email_for_retraining, label_input],
            outputs=retrain_result
        )

        gr.Markdown("## 📘 Glossary and Explanation of Labels")
        gr.Markdown(
            """
            ### Labels:
            - **Spam:** Unwanted or harmful emails flagged by the system.
            - **Ham:** Legitimate, safe emails.

            ### Confusion Matrix:
            The confusion matrix shows the performance of the model by comparing the true labels with the predicted ones. 
            It consists of:
            - **True Positives (TP):** Correctly predicted spam emails.
            - **True Negatives (TN):** Correctly predicted ham emails.
            - **False Positives (FP):** Ham emails incorrectly predicted as spam.
            - **False Negatives (FN):** Spam emails incorrectly predicted as ham.

            ### Metrics:
            - **Accuracy:** The percentage of correct classifications.
            - **Precision:** Out of predicted Spam, how many are actually Spam.
            - **Recall:** Out of all actual Spam emails, how many are predicted as Spam.
            - **F1 Score:** Harmonic mean of Precision and Recall.
            """
        )

    return interface

# Launch the interface
interface = create_interface()
interface.launch(share=True)

