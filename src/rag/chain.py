from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
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
            question_generator_chain_kwargs={"prompt": CONDENSE_QUESTION_PROMPT},
            verbose=self.config['rag']['chain']['verbose']
        )
        
    def query(self, question: str, chat_history=None):
        """Process a question through the RAG chain."""
        if chat_history is None:
            chat_history = []
            
        try:
            response = self.chain({"question": question, "chat_history": chat_history})
            return {
                "answer": response["answer"],
                "source_documents": response["source_documents"],
                "chat_history": self.memory.chat_memory.messages
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise