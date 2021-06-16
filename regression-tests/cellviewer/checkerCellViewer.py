import sys
import argparse
import json
import os
from svgwrite import Drawing
import re
from config import *

# helper function to sort dictionaries as tuples
def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else: return str(obj)

# helper function to revert back to dictionary
def revert(obj):
    if isinstance(obj, tuple):
        return {obj[0]:revert(obj[1])}
    if isinstance(obj, list):
        if isinstance(obj[0], tuple):
            return {k:revert(v) for k,v in obj}
        else:
            return [revert(x) for x in obj]
    else: return obj
    
def isfloat(obj):
    try:
        float(obj)
        return True
    except ValueError:
        return False

# helper function to compare ordered list
def isequal(obj,obj2):
    if type(obj) != type(obj2): 
        return False
    if isinstance(obj,list) or isinstance(obj,tuple):
        if len(obj)!=len(obj2): return False
        for i in range(len(obj)): 
            if not isequal(obj[i],obj2[i]): return False
        return True
    elif isinstance(obj, str) and isfloat(obj) and isfloat(obj2):
        return isequal(float(obj), float(obj2))
    elif type(obj)==float:
        return abs(obj - obj2) <= 1e-09 * max(abs(obj), abs(obj2)) # check if floats are same within 9 digits
    else:
        return obj==obj2

# checks for line differences
def lineCheck(testLines, expectedLines):
    
    extraLines = [{     "type":"Extra Line", 
                        "key" :"Line "+str(i), 
                        "test":testLines[i], 
                        "expected":""}
                        for i in range(len(expectedLines),len(testLines))]

    missingLines = [{   "type":"Missing Line", 
                        "key" :"Line "+str(i), 
                        "test":"", 
                        "expected":expectedLines[i]} 
                        for i in range(len(testLines),len(expectedLines))]
    
    diffLines = [{      "type":"Different Lines", 
                        "key" :"Line "+str(i), 
                        "test":testLines[i], 
                        "expected":expectedLines[i]} 
                        for i in range(min(len(testLines),len(expectedLines)))
                        if testLines[i]!=expectedLines[i]]
    return diffLines + extraLines + missingLines

# check for value differences
def valueCheck(test, expected):
    i=j=0
    diffValues = []
    extraValues = []
    missingValues = []
    while i<len(test) and j<len(expected):
        if not isequal(test[i], expected[j]):
            if test[i][0] < expected[j][0]:
                extraValues.append(test[i])
                j-=1
            elif test[i][0] > expected[j][0]:
                missingValues.append(expected[j])
                i-=1
            else:
                diffValues.append({      
                    "type":"Different Values", 
                    "key" :str(test[i][0]), 
                    "test":json.dumps(revert(test[i]), indent=4),
                    "expected":json.dumps(revert(expected[j]), indent=4)
                })
        i+=1
        j+=1
                
    while i<len(test):
        extraValues.append(test[i])
    while j<len(expected):
        missingValues.append(expected[i])
        
    for i in range(len(extraValues)):
        diffValues.append({      
            "type":"Extra Value", 
            "key" :str(extraValues[i][0]), 
            "test":json.dumps(revert(extraValues[i]), indent=4),
            "expected":""
        })
        for j in range(len(missingValues)):
            if isequal(extraValues[i], missingValues[j]):
                missingValues.pop(j);
                diffValues.pop()
                break;
                
    for i in range(len(missingValues)):
        diffValues.append({      
            "type":"Missing Value",
            "key" :str(missingValues[i][0]), 
            "test":"",
            "expected":json.dumps(revert(missingValues[i]), indent=4)
        })

    return diffValues

def checkJSON(test_path, expected_path, lvl, file):

    test_path += file
    expected_path += file
    testLines = []
    expectedLines = []

    try:
        with open(test_path,'r') as tFile:
            testData = tFile.read()
            test = ordered(json.loads(testData))
            testLines = testData.split('\n')

    except IOError as e:
        err = 'Test JSON: An IOError occurred. '+e.args[-1]+'. File: '+test_path
        print(err)
        return []
    
    try:
        with open(expected_path,'r') as eFile:
            expectedData = eFile.read()
            expected = ordered(json.loads(expectedData))
            expectedLines = expectedData.split('\n')

    except IOError as e:
        err = 'Expected JSON: An IOError occurred. '+e.args[-1]+'. File: '+expected_path
        print(err)
        return []

    lineDiff = lineCheck(testLines, expectedLines) if not lvl%2 else None
    valDiff = valueCheck(test, expected) if lvl else None
    
    return [valDiff, lineDiff]
    
def checkSVG(test_path, expected_path, lvl, file):
    test_path += file
    expected_path += file

    try:
        with open(test_path,'r') as tFile:
            tData = tFile.read().split("<")
            tData = [tag.split() for tag in tData]
            tData = sorted([sorted([sorted(attr.split(",")) for attr in tag]) for tag in tData])
            tData = [tag for tag in tData if isinstance(tag, list) and len(tag) > 0]
    
    except IOError as e:
        err = 'Test SVG: An IOError occurred. '+e.args[-1]+'. File: '+test_path
        print(err)
        return []

    try:
        with open(expected_path,'r') as eFile:
            eData = eFile.read().split("<")
            eData = [tag.split() for tag in eData]
            eData = sorted([sorted([sorted(attr.split(",")) for attr in tag]) for tag in eData])
            eData = [tag for tag in eData if isinstance(tag, list) and len(tag) > 0]

    except IOError as e:
        err = 'Expected SVG: An IOError occurred. '+e.args[-1]+'. File: '+expected_path
        print(err)
        return []     

    return [valueCheck(tData, eData)]


def main():
    
    parser = argparse.ArgumentParser(description='Regression Test for Cell Viewer')
    parser.add_argument('-wd', metavar='WORKING_DIRECTORY', type=str,
                        help='the path to working directory', default=".") 
    parser.add_argument('-test_path', metavar='TEST_PATH', type=str,
                        help='the path to test data', default="test/") 
    parser.add_argument('-expected_path', metavar='EXPECTED_PATH', type=str,
                        help='the path to expected results', default="expected/") 
    parser.add_argument('-output_path', metavar='OUTPUT_PATH', type=str,
                        help='the desired output path', default="results/") 
    parser.add_argument('-angle', metavar='CHECK_ANGLE', type=bool,
                        help='check angle?', default=True) 
    parser.add_argument('-colony', metavar='CHECK_COLONY', type=bool,
                        help='check colony?', default=True) 
    parser.add_argument('-pie', metavar='CHECK_PIE', type=bool,
                        help='check pie?', default=True) 
    parser.add_argument('-tree', metavar='CHECK_TREE', type=bool,
                        help='check tree?', default=True) 
    parser.add_argument('-checkLevel', metavar='CHECK_LEVEL', type=int,
                        help='lines only: 0, values only: 1, both: 2', default=2) 
    args = parser.parse_args()       

    test = args.wd+"/"+args.test_path
    expected = args.wd+"/"+args.expected_path

    if test[-1]!='/': test+='/'
    if expected[-1]!='/': expected+='/'

    results = []
    if args.angle:  results.append(["Angle Check"]+checkJSON(test, expected, args.checkLevel, angleFilename))
    if args.colony: results.append(["Colony Check"]+checkJSON(test, expected, args.checkLevel, colonyFilename))
    if args.pie:    results.append(["Pie Check"]+checkSVG(test, expected, args.checkLevel, pieFilename))
    if args.tree:   results.append(["Tree Check"]+checkSVG(test, expected, args.checkLevel, treeFilename))

    out = open(args.wd+"/"+args.output_path+resultFilename, 'w')
    for testResult in results:
        out.write('===================================================================\n')
        out.write(testResult[0].upper()+'\n')
        out.write('===================================================================\n')
        for tRes in range(1,len(testResult)):
            if testResult[tRes]:
                out.write('Test ' + str(tRes) + ' Failed\n')
                out.write('------------------------------------------------------------\n')
                for diff in testResult[tRes]:
                    out.write(diff["type"]+" at "+diff["key"]+':\n')  
                    out.write('TEST:\n')
                    out.write(diff["test"]+'\n') 
                    out.write('EXPECTED:\n') 
                    out.write(diff["expected"]+'\n')   
                    out.write('------------------------------------------------------------\n')
            else:
                out.write('Test '+str(tRes)+' Passed\n')  
    out.close()  

if __name__ == '__main__':
    main()
