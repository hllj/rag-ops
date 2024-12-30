from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import langchain.chat_models
import langchain.chat_models.openai
from langchain.memory import ConversationBufferMemory
from .prompt_templates import QA_PROMPT, CONDENSE_QUESTION_PROMPT
from .retrievers import DocumentRetriever
import yaml
import logging

class RAGChain:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.retriever = DocumentRetriever(config_path)
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # Create the chain
        self.chain = self._create_chain()
        
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        if self.config['llm']['provider'] == "openai":
            return ChatOpenAI(
                model_name=self.config['llm']['model_name'],
                temperature=self.config['llm']['temperature'],
                max_tokens=self.config['llm']['max_tokens']
            )
        # Add support for other providers as needed
        
    def _create_chain(self):
        """Create the RAG chain based on configuration."""
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever.retriever,
            memory=self.memory,
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
            # Extract answer and source documents explicitly
            answer = response.get("answer", "No answer found.")
            source_documents = response.get("source_documents", [])
            return {
                "answer": answer,
                "source_documents": source_documents,
                "chat_history": self.memory.chat_memory.messages
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise