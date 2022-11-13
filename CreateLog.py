# part of https://github.com/JakobPCoder/TexModTools 
# only use according to the license


import os
import pathlib

LOG_FILE_NAME = "abc.log"
INPUT_FILE_FORMAT = ".png"



basePath = os.path.dirname(os.path.realpath(__file__))
with os.scandir(basePath) as i:   
    allEntriesLogs = ""
    with open(basePath + "/" + LOG_FILE_NAME, 'w') as filetowrite: 
        for entry in i:
            if entry.is_file():
                path = pathlib.Path(entry)
                fileName = path.name
                suffix = path.suffix       
                    
                if suffix == INPUT_FILE_FORMAT:
                    pathStr = str(path)
                    
                    endIndex = pathStr.rfind("_") + 1
                    address = pathStr[endIndex:-len(INPUT_FILE_FORMAT)]
                    
                    logLine = address + "|" + pathStr + "\n"
                    allEntriesLogs += logLine
                    print(logLine)
                    
        filetowrite.write(allEntriesLogs)
