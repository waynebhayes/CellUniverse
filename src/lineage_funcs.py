from cell import Bacilli
import optimization
from typing import List
from global_optimization.Modules import CellNodeM, LineageM
import pandas as pd
from colony import LineageFrames, CellNode
import csv
import numpy as np


def build_initial_lineage(imagefiles, realimages, lineagefile, continue_from, simulation_config):
    # create a lineage given the path of the lineagefile and requiring imagefiles. The output lineage will contain frame number up to the continue_from(include)
    cells_data = pd.read_csv(lineagefile)
    cells_data = cells_data.replace('None', None)
    lineage = LineageM(simulation_config)
    for i in range(len(imagefiles)):
        filename = imagefiles[i].name
        # this is some what a ugly way to find out frame number contained in a string. Should be improved later?
        current_frame_number = int(filename.split('.')[0][-3:])
        if current_frame_number > continue_from:
            break
        if i > 0:
            lineage.forward()
        for _, row in cells_data[cells_data["file"] == filename].iterrows():
            new_cell = Bacilli(row["name"], row["x"], row["y"], row["width"], row["length"], row["rotation"], row["split_alpha"], row["opacity"])
            lineage.frames[-1].add_cell(new_cell)
        lineage.frames[i].simulation_config = optimization.find_optimal_simulation_conf(lineage.frames[i].simulation_config, realimages[i], lineage.frames[i].nodes)
    return lineage

def load_colony(colony, initial_file, config, initial_frame=None):
    """Loads the initial colony of cells."""
    with open(initial_file, newline='') as fp:
        reader = csv.DictReader(fp, skipinitialspace=True)
        for row in reader:
            if initial_frame is not None and row['file'] != initial_frame:
                continue
            name = row['name']
            celltype = config['global.cellType'].lower()
            if celltype == 'bacilli':
                x = float(row['x'])
                y = float(row['y'])
                width = float(row['width'])
                length = float(row['length'])
                rotation = float(row['rotation'])
                if config["simulation"]["image.type"] == "graySynthetic":
                    opacity = config["simulation"]["cell.opacity"]
                    cell = Bacilli(name, x, y, width, length, rotation, opacity=opacity)
                else:
                    cell = Bacilli(name, x, y, width, length, rotation)
            colony.add(CellNode(cell))


def create_lineage(imagefiles, realimages, config, args):
    lineageframes = LineageFrames()
    colony = lineageframes.forward()
    if args.lineage_file:
        load_colony(colony, args.lineage_file, config, initial_frame=imagefiles[0].name)
        lineage = build_initial_lineage(imagefiles, realimages, args.lineage_file, args.continue_from, config["simulation"])
    else:
        load_colony(colony, args.initial, config)
        lineage = build_initial_lineage(imagefiles, realimages, args.initial, args.continue_from, config["simulation"])
    return lineage


def save_lineage(filename, cellnodes: List[CellNodeM], lineagefile):
    for node in cellnodes:
        properties = [filename, node.cell.name]
        properties.extend([
            str(node.cell.x),
            str(node.cell.y),
            str(node.cell.width),
            str(node.cell.length),
            str(node.cell.rotation),
            str(node.cell.split_alpha),
            str(node.cell.opacity)])
        print(','.join(properties), file=lineagefile)
