from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("i'm happy"))