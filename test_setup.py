from transformers import pipeline
clf = pipeline('sentiment-analysis',
  model='distilbert-base-uncased-finetuned-sst-2-english')
print(clf('Hugging Face is amazing!'))
