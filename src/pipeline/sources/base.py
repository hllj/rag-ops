from abc import ABC, abstractmethod
from queue import Queue
import logging
from typing import Optional

class SourceHandler(ABC):
    """Base class for handling different document sources."""
    def __init__(self, config: dict):
        self.config = config
        self.processing_queue: Optional[Queue] = None
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def start(self) -> None:
        """Start monitoring the source for new documents."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop monitoring the source."""
        pass

    def set_processing_queue(self, queue: Queue) -> None:
        """Set the queue where new documents should be sent for processing."""
        self.processing_queue = queue

    def _validate_queue(self) -> None:
        """Validate that processing queue has been set."""
        if self.processing_queue is None:
            raise ValueError("Processing queue has not been set")