"""Cell Universe Post-Processor 2

Converts colony.cvs from Cell Universe to input 
for radialtree.py"""

import argparse
from pathlib import Path

def main():
    # Get the path to the colony data from the command-line
    parser = argparse.ArgumentParser(
        description='Converts Cell Universe data to be read to create a radial tree plot.')
    parser.add_argument('colony_path', metavar='DATA_PATH', type=Path,
                        help='the path to colony data (in CSV format)')
    parser.add_argument('output_path', metavar='OUTPUT_DATA', type=Path,
                        help='the path to the outputted data (in CSV format)')

    args = parser.parse_args()

    f = open(args.colony_path, 'r')
    data = [line.strip().split(',') for line in f]
    headers = {data[0][k]:k for k in range(len(data[0]))}
    out = "ImageNumber,ObjectID,ParentObjectID\n"
    data.pop(0)
    
    cells = {}
    
    for cell in data:
        frame=int(cell[headers["frame"]].split('.')[0].split('frame')[1])+1
        out+=str(frame)+','
        
        name = cell[headers["name"]]
        out+=name[1:]+','
        
        if frame>1: out+=name[1:-1]
        else: out+='0'
        if name not in cells: cells[name] = 1
        else: out+=name[-1]            
        out+="\n"
    
    f.close()
    
    f = open(args.output_path, 'w')
    f.write(out)
    
if __name__ == '__main__':
    main()