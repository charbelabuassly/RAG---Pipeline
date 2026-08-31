#Responsible for reading TXT type files



class TXTReader:
    
    def read(self, path):
        with open(path) as f:
            return f.read()