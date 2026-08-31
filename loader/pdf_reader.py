#Responsible for reading PDF type files
from pypdf import PdfReader


class PDFReader:
    
    def read(self,path):
        reader = PdfReader(path)
        pages = [] # each entry in here will store and extracted page from the pdf
        
        for page in reader.pages:
            text = page.extract_text()
            
            pages.append(text)
            
        return '\n'.join(pages)