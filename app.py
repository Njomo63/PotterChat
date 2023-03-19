import os
import datetime
import pickle

import gradio as gr
import langchain
from langchain.llms import HuggingFaceHub

from chain import get_new_chain

api_token = os.environ["HF_TOKEN"]


def get_faiss_store():
    with open("docs.pkl", "rb") as f:
        faiss_store = pickle.load(f)
        return faiss_store

def load_model():

    print(langchain.__file__)

    vectorstore = get_faiss_store()

    flan_ul = HuggingFaceHub(repo_id="google/flan-ul2", 
                                model_kwargs={"temperature":0.1, "max_new_tokens":200},
                                huggingfacehub_api_token=api_token)

    qa_chain = get_new_chain(vectorstore, flan_ul)
    return qa_chain


def chat(inp, agent):
    result = []
    if agent is None:
        result.append((inp, "Please wait for model to load (3-5 seconds)"))
        return result
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("inp: " + inp)
    result = []
    output = agent({"question": inp})
    answer = output["answer"]
    result.append((inp, answer))
    print(result)
    return result


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>PotterChat</center></h3><p>Ask questions about the Harry Potter Books, Powered by Flan-UL2</p>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Who was Harry's godfather?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "Which house in Hogwarts was Harry in?",
            "Who were Harry's best friends?",
            "Who taught Potions at Hogwarts?",
        ],
        inputs=message,
    )

    gr.HTML(
        """
    This simple application uses Langchain, an open-source LLM, and FAISS to do Q&A over the Harry Potter books."""
    )

    gr.HTML(
        "<center>Powered by <a href='huggingface.co'>Hugging Face 🤗</a> and <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    # state = gr.State()
    agent_state = gr.State()

    block.load(load_model, inputs=None, outputs=[agent_state])

    # submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    # message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    submit.click(chat, inputs=[message, agent_state], outputs=[chatbot])
    message.submit(chat, inputs=[message, agent_state], outputs=[chatbot])

block.launch(debug=True)