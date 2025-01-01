import logging
import queue

from base import IngestionSource

class FolderWatcherSource(IngestionSource):
    """Folder watcher implementation for an ingestion source."""
    def __init__(self, watch_directory: str, logger: logging.Logger):
        self.watch_directory = watch_directory
        self.logger = logger
        self.observer = None

    def start(self, processing_queue: queue.Queue):
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class FileEventHandler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    processing_queue.put(event.src_path)

        self.handler = FileEventHandler()
        self.observer = Observer()
        self.observer.schedule(self.handler, self.watch_directory, recursive=False)
        self.observer.start()
        self.logger.info(f"Started watching directory: {self.watch_directory}")

    def stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()