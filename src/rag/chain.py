import yaml
import logging
from typing import List, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import langchain.chat_models
import langchain.chat_models.openai
from langchain.memory import ConversationBufferMemory

from .prompt_templates import QA_PROMPT, CONDENSE_QUESTION_PROMPT
from .retrievers import DocumentRetriever
from ..evaluation.rag_evaluation import RAGEvaluator

class RAGChain:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.retriever = DocumentRetriever(config_path)
        self.llm = self._initialize_llm()
        
        # Create the chain
        self.chain = self._create_chain()
        self.evaluator = RAGEvaluator()

    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        if self.config['llm']['provider'] == "openai":
            return ChatOpenAI(
                model_name=self.config['llm']['model_name'],
                temperature=self.config['llm']['temperature'],
                max_tokens=self.config['llm']['max_tokens']
            )

    def _create_chain(self):
        """Create the RAG chain based on configuration."""
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever.retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            verbose=self.config['rag']['chain']['verbose']
        )

    def query(self, question: str, chat_history=None):
        """Process a question through the RAG chain."""
        if chat_history is None:
            chat_history = []
            
        try:
            response = self.chain({"question": question, "chat_history": chat_history})
            return {
                "answer": response.get("answer", "No answer found."),
                "source_documents": response.get("source_documents", [])
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise