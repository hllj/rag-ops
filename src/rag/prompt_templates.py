from langchain.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI assistant. Use the following context to answer the question. 
    If you don't know the answer, just say you don't know. Don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer: """
)

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the following conversation and a follow up question, rephrase the follow up question 
    to be a standalone question that captures all relevant context from the chat history.

    Chat History: {chat_history}
    Follow Up Question: {question}

    Standalone Question:"""
)