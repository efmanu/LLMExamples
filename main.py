from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
from dotenv import load_dotenv

load_dotenv()


def print_sentiment():
    # Use a breakpoint in the code line below to debug your script.
    print(f"Sentiment Analysis")  # Press ⌘F8 to toggle the breakpoint.
    print(pipeline("sentiment-analysis")("we love you"))


def print_speech_to_text() -> str:
    generator = pipeline(task="automatic-speech-recognition")
    speech = generator("https://listenaminute.com/a/actors.mp3")
    print(f"Speech:\n {speech}")
    return speech


def identify_subject_with_pipeline_llm(speech: str):
    model_id = "google/flan-t5-large"

    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, config=config)
    pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_length=512
    )

    hf_llm = HuggingFacePipeline(pipeline=pipe)

    template = """Identify the subject of the following speech: {speech}

        Subject: """
    prompt = PromptTemplate(template=template, input_variables=["speech"])
    llm_chain = LLMChain(prompt=prompt, llm=hf_llm)
    return llm_chain.run(speech)


def identify_subject_with_pipeline(speech: str) -> str:
    repo_id = "google/flan-t5-large"
    llm = HuggingFacePipeline.from_model_id(
        model_id=repo_id,
        task="text2text-generation",
        model_kwargs={"temperature": 0, "max_length": 64},
    )

    template = """Identify the subject of the following speech: {speech}

    Subject: """
    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm

    return chain.invoke({"speech": speech})


def identify_subject_with_hub(speech: str) -> str:
    template = """Identify the subject of the following speech: {speech}

        Subject: """

    prompt = PromptTemplate(template=template, input_variables=["speech"])
    repo_id = "google/flan-t5-xxl"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain.run(speech)


if __name__ == "__main__":
    print_sentiment()
    speech = print_speech_to_text()
    print(f"Pipeline:\n")
    print(identify_subject_with_pipeline(speech["text"]))
    print(f"Hub:\n")
    print(identify_subject_with_hub(speech["text"]))
    print(f"Pipeline Chain:\n")
    print(identify_subject_with_pipeline_llm(speech["text"]))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
