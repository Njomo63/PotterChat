from typing import Dict, List

from langchain import PromptTemplate
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from pydantic import BaseModel


class CustomChain(Chain, BaseModel):

    vstore: FAISS
    chain: BaseCombineDocumentsChain

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    # def _call(self, inputs: str) -> Dict[str, str]:
        question = inputs["question"]
        # question = inputs
        docs = self.vstore.similarity_search(question, k=5)
        answer, _ = self.chain.combine_docs(docs, **inputs)
        return {"answer": answer}
    

def get_new_chain(vectorstore, llm):
    flan_template = """Use only the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(template=flan_template, input_variables=["question", "context"])

    doc_chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=PROMPT,
        verbose=True
        )
    return CustomChain(chain=doc_chain, vstore=vectorstore)