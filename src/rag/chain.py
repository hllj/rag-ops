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
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
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
        
    @property
    def chat_history(self) -> List[Dict[str, str]]:
        """Convert chat memory to list of messages."""
        messages = self.memory.chat_memory.messages
        return [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in messages
        ]
        
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
                "chat_history": self.chat_history
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise
    
    def evaluate(self, questions: List[str]) -> Dict:
        """
        Evaluate RAG pipeline performance using Ragas metrics.
        
        Args:
            questions: List of questions to evaluate
            
        Returns:
            Dict containing evaluation scores
        """
        try:
            # Get responses for all questions
            answers = []
            contexts = []
            
            for question in questions:
                response = self.query(question)
                answers.append(response["answer"])
                
                # Extract context texts from source documents
                question_contexts = [doc.page_content for doc in response["source_documents"]]
                contexts.append(question_contexts)
                
            # Run evaluation
            scores = self.evaluator.evaluate(
                questions=questions,
                contexts=contexts,
                answers=answers
            )
            
            return scores
            
        except Exception as e:
            logging.error(f"Error during RAG evaluation: {str(e)}")
            raise