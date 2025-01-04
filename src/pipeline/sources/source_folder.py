from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
from typing import Set

from .base import SourceHandler

class FolderWatchHandler(FileSystemEventHandler):
    """Handles file system events for the folder watcher."""
    def __init__(self, source_handler: 'FolderSourceHandler'):
        self.source_handler = source_handler
        self.supported_formats = source_handler.supported_formats

    def on_created(self, event):
        """Handle file creation event."""
        if not event.is_directory and self.source_handler._is_supported_format(event.src_path):
            self.source_handler.handle_new_file(event.src_path)

class FolderSourceHandler(SourceHandler):
    """Handles watching a folder for new documents."""
    def __init__(self, config: dict, watch_directory: str):
        super().__init__(config)
        self.watch_directory = watch_directory
        self.supported_formats = set(config['document_processor']['supported_formats'])
        self.observer = Observer()
        self.watch_handler = FolderWatchHandler(self)

    def start(self) -> None:
        """Start watching the directory."""
        self._validate_queue()
        self.observer.schedule(self.watch_handler, self.watch_directory, recursive=False)
        self.observer.start()
        self.logger.info(f"Started watching directory: {self.watch_directory}")
        
        # Process existing files
        self._process_existing_files()

    def stop(self) -> None:
        """Stop watching the directory."""
        self.observer.stop()
        self.observer.join()
        self.logger.info("Folder watcher stopped")

    def handle_new_file(self, file_path: str) -> None:
        """Handle a new file detected in the watch directory."""
        self.logger.info(f"New file detected: {file_path}")
        self._validate_queue()
        self.processing_queue.put({"source": "folder", "path": file_path})

    def _process_existing_files(self) -> None:
        """Process any existing files in the watch directory."""
        for filename in os.listdir(self.watch_directory):
            file_path = os.path.join(self.watch_directory, filename)
            if os.path.isfile(file_path) and self._is_supported_format(file_path):
                self.handle_new_file(file_path)

    def _is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported."""
        extension = os.path.splitext(file_path)[1][1:].lower()
        return extension in self.supported_formats