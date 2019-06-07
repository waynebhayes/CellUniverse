"""Cell Universe Post-Processor 2

Converts colony.csv from Cell Universe to input 
for radialtree.py and Cell Viewer"""

from config import *

def parseColony(pathname):

    # read result from Cell Universe
    f = open(pathname+"/"+lineageFilename, 'r')
    data = [line.strip().split(',') for line in f]
    f.close()
    headers = {data[0][k]:k for k in range(len(data[0]))}
    out = []
    data.pop(0)

    colony=[]
    cells = {}
    frame = [0,""]

    # processing file
    for cell in data:
        
        line = {}

        # ImageNumber, frameName
        if frame[1]!=cell[headers["file"]]:
            frame[0]+=1
            frame[1]=cell[headers["file"]]
            colony.append({
                "frame": frame[1],
                "cells": []
            })
        line["ImageNumber"]=str(frame[0])
        
        # ObjectID
        name = cell[headers["name"]]
        line["ObjectID"]=name[1:]
        
        # ParentObjectID
        if frame[0]>1: line["ParentObjectID"]=name[1:-1]
        else: line["ParentObjectID"]=''
        if name not in cells: cells[name] = 1
        else: line["ParentObjectID"]+=name[-1]
        out.append(line)

        # Colony
        colony[-1]["cells"].append([
            "\""+cell[headers["name"]]+"\"",
            cell[headers["x"]],
            cell[headers["y"]]
        ])
    

    # output colony json file
    fout = open(pathname+"/"+colonyFilename, 'w')
    fout.write(
        "{\n\t"+
            "\"frames\": "+str(frame[0])+",\n\t"
            "\"frameNumber\": {\n\t\t"+
                "\"frame\": \"frameName\",\n\t\t"+
                "\"cells\": [\n\t\t\t"+
                    "[\"name\",\"x\",\"y\"]\n\t\t"+
                "]\n\t"
            "}")
    for i in range(frame[0]):
        outframe = (",\n\t"+
                "\""+str(i)+"\": {\n\t\t"+
                    "\"frame\": \""+colony[i]["frame"]+"\",\n\t\t"+
                    "\"cells\": [")
        
        for cell in colony[i]["cells"]:
            outframe += "\n\t\t\t["+cell[0]+","+cell[1]+","+cell[2]+"],"
        
        fout.write(outframe[:-1]+"\n\t\t]\n\t}")
            
    fout.write("\n}")
    fout.close()

    return out
    
if __name__ == '__main__':
    # write to output file
    f = open("processedColony.csv", 'w')
    f.write("ImageNumber,ObjectID,ParentObjectID\n")
    for line in parseColony("."):
        f.write(line["ImageNumber"]+","+line["ObjectID"]+","+line["ParentObjectID"]+"\n")
    f.close()