import os
import logging
import queue

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from base import DataSource

class FolderWatcherDataSource(FileSystemEventHandler, DataSource):
    """Folder watcher implementation for an ingestion source."""
    def __init__(self, directory: str, supported_formats: list, logger: logging.Logger):
        self.directory = directory
        self.supported_formats = supported_formats
        self.callback = None
        self.observer = None
        self.logger = logger

    def start(self, callback):
        """Start watching the directory."""
        self.callback = callback
        self.observer = Observer()
        self.observer.schedule(self, self.directory, recursive=False)
        self.observer.start()
        self.logger.info(f"Started watching directory: {self.directory}")
        self._process_existing_files()

    def stop(self):
        """Stop watching the directory."""
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            file_path = event.src_path
            if self._is_supported_format(file_path):
                self.callback(file_path)
                
    def _process_existing_files(self):
        """Process any existing files in the watch directory."""
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            if os.path.isfile(file_path) and self._is_supported_format(file_path):
                self.logger.info(f"Found existing file: {filename}")
                self.callback(file_path)

    def _is_supported_format(self, file_path: str) -> bool:
        extension = os.path.splitext(file_path)[1][1:].lower()
        return extension in self.supported_formats