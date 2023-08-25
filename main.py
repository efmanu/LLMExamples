from transformers import pipeline
from transformers import pipeline


def print_sentiment():
    # Use a breakpoint in the code line below to debug your script.
    print(f"Sentiment Analysis")  # Press ⌘F8 to toggle the breakpoint.
    print(pipeline("sentiment-analysis")("we love you"))


def print_speech_to_text():
    generator = pipeline(task="automatic-speech-recognition")
    print(generator("https://listenaminute.com/a/actors.mp3"))


if __name__ == "__main__":
    print_sentiment()
    print_speech_to_text()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
