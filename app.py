```python
import gradio as gr
import pandas as pd
from transformers import pipeline

# Load model
classifier = pipeline(
    "text-classification",
    model="./my_model"
)

def analyze_csv(file):

    df = pd.read_csv(file.name)

    sentiments = []

    for text in df["text"]:
        result = classifier(str(text))
        sentiments.append(result[0]["label"])

    df["Sentiment"] = sentiments

    positive = (df["Sentiment"] == "LABEL_1").sum()
    negative = (df["Sentiment"] == "LABEL_0").sum()

    summary = f"""
Positive Feedback: {positive}

Negative Feedback: {negative}
"""

    return df, summary

demo = gr.Interface(
    fn=analyze_csv,
    inputs=gr.File(label="Upload CSV File"),
    outputs=[
        gr.Dataframe(label="Analysis Results"),
        gr.Textbox(label="Summary")
    ],
    title="📊 Student Feedback Analyzer",
    description="Upload a CSV file containing a 'text' column."
)

demo.launch()
```
