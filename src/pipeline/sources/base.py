import queue

class IngestionSource:
    """Base class for ingestion sources."""
    def start(self, processing_queue: queue.Queue):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError