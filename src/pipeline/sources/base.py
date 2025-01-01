from abc import ABC, abstractmethod

class DataSource(ABC):
    @abstractmethod
    def start(self, callback):
        """Start the data source and feed data to the provided callback."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the data source."""
        pass