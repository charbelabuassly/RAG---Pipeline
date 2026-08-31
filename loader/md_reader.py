#Responsible for reading Markdown type files



class MarkdownReader:

    def read(self, path):
        with open(path, "r", encoding="utf-8") as file:
            return file.read()