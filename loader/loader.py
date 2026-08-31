from pathlib import Path

from .txt_reader import TXTReader
from .pdf_reader import PDFReader
from .md_reader import MarkdownReader


class DocumentLoader:

    def __init__(self):
        self.readers = {
            ".txt": TXTReader(),
            ".pdf": PDFReader(),
            ".md": MarkdownReader()
        }

    def load(self, path):

        path = Path(path)
        extension = path.suffix.lower()
        if extension not in self.readers:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )
        reader = self.readers[extension]

        return reader.read(path)