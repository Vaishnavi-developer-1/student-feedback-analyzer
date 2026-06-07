from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./my_model"
)

while True:

    text = input(
        "\nEnter Feedback (type exit to quit): "
    )

    if text.lower() == "exit":
        break

    result = classifier(text)

    prediction = result[0]["label"]

    if prediction == "LABEL_1":
        print("Positive Feedback")
    else:
        print("Negative Feedback")

    print(result)