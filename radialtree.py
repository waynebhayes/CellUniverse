"""Converts SuperSegger Cell Tracking data into a radial tree plot."""

import argparse
import csv
import math
import sys
from pathlib import Path

from svgwrite import Drawing

SVG_SIZE = 512
STEP_LENGTH = math.floor(SVG_SIZE/2/81)


def _print_error(msg):
    print(f'ERROR: {msg}', file=sys.stderr)


class CellNode(object):
    '''Object representing an instance of a single cell over time.'''

    def __init__(self, object_id, parent_id, initial_frame):
        self._object_id = object_id
        self._parent_id = parent_id
        self._start_frame = initial_frame
        self._end_frame = initial_frame + 1
        self._children = []

        self.angle = 0.0

    def exists_in(self, frame):
        '''Set the latest frame the cell exists in.'''
        self._end_frame = frame + 1

    @property
    def object_id(self):
        return self._object_id

    @property
    def parent_id(self):
        return self._parent_id

    @property
    def start_frame(self):
        return self._start_frame

    @property
    def end_frame(self):
        return self._end_frame

    @property
    def children(self):
        return self._children


def angle_spacing_generator(count):
    spacing = 2*math.pi/count
    for i in range(count):
        yield i*spacing


def get_leaf_count(node):
    '''Retruns the number of leaf nodes the plot has in total.'''
    if not node.children:
        return 1
    return sum([get_leaf_count(child) for child in node.children])


def set_angles(node, angle_spacings):
    '''Sets the angles for all of the nodes.'''
    if not node.children:
        node.angle = next(angle_spacings)
    else:
        angles = [set_angles(child, angle_spacings) for child in node.children]
        node.angle = sum(angles)/len(node.children)
    return node.angle


def convert_to_tree(rows):
    '''Returns a list of the root nodes.'''
    merge_dict = {}
    root_list = []

    for row in rows:
        frame = int(row['ImageNumber'])
        object_id = int(row['ObjectID'])
        parent_id = int(row['ParentObjectID'])

        if object_id in merge_dict:
            merge_dict[object_id].exists_in(frame)
        else:
            node = CellNode(object_id, parent_id, frame)

            if parent_id != 0:
                merge_dict[parent_id].exists_in(frame-1)
            merge_dict[object_id] = node

            if parent_id == 0:
                root_list.append(node)
            else:
                merge_dict[parent_id].children.append(node)

    return root_list


def compress_edge(root):
    # find the furthest the individual cell goes
    node = root.children[0]
    while len(node.children) == 1:
        node = node.children[0]

    # compress
    if not node.children:
        root.exists_in(node.end_frame)
        root.children.clear()
    else:
        root.exists_in(node.children[0].start_frame - 1)
        root.children.clear()
        root.children.extend(node.children)


def compress_tree(node):
    if len(node.children) == 1:
        compress_edge(node)
    for child in node.children:
        compress_tree(child)


def draw_radial_tree_node(svg_drawing, tree_plot, node):
    tree_plot.add(svg_drawing.line(
        start=(STEP_LENGTH*node.start_frame, 0),
        end=(STEP_LENGTH*node.end_frame, 0),
        transform=f'translate({SVG_SIZE/2} {SVG_SIZE/2}) '
                  f'rotate({-180*node.angle/math.pi})'))

    if node.children:
        for child in node.children:
            draw_radial_tree_node(svg_drawing, tree_plot, child)

        angles = [child.angle for child in node.children]
        max_angle = max(angles)
        min_angle = min(angles)

        path = svg_drawing.path(
            f'M{STEP_LENGTH*node.end_frame*math.cos(-max_angle)},'
            f'{STEP_LENGTH*node.end_frame*math.sin(-max_angle)}',
            transform=f'translate({SVG_SIZE/2} {SVG_SIZE/2})')
        path.push_arc(
            (STEP_LENGTH*node.end_frame*math.cos(-min_angle),
             STEP_LENGTH*node.end_frame*math.sin(-min_angle)),
            0,
            (STEP_LENGTH*node.end_frame, STEP_LENGTH*node.end_frame),
            large_arc=False,
            absolute=True)
        tree_plot.add(path)


def save_radial_tree_plot(filename, root_list):
    svg_drawing = Drawing(
        filename=filename,
        size=(SVG_SIZE, SVG_SIZE),
        debug=True)

    tree_plot = svg_drawing.add(svg_drawing.g(
        id='treeplot',
        style='stroke: black; stroke-width: 1; fill: none; stroke-linecap: round;'))

    for root in root_list:
        draw_radial_tree_node(svg_drawing, tree_plot, root)

    svg_drawing.save()


def main():
    # Get the path to the lineage data from the command-line
    parser = argparse.ArgumentParser(description='Generate a radial tree plot.')
    parser.add_argument('path', metavar='LINEAGE_DATA', type=Path,
                        help='the path to the lineage data (in CSV format)')
    parser.add_argument('svg_path', metavar='SVG_FILE', type=Path,
                        help='the path to the outputted plot')

    args = parser.parse_args()

    if not args.path.exists():
        print(f'File not found: {args.path.absolute()}')
        return

    # Load the data into memory
    rows = []
    with args.path.open() as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            rows.append(row)

    # Merge individual cells temporally and generate tree
    root_list = convert_to_tree(rows)
    leaf_counts = [get_leaf_count(root) for root in root_list]
    angle_spacings = angle_spacing_generator(sum(leaf_counts))
    for root in root_list:
        set_angles(root, angle_spacings)

    # Compress tree
    for root in root_list:
        compress_tree(root)

    # Draw the SVG radial tree plot
    save_radial_tree_plot(args.svg_path, root_list)


if __name__ == '__main__':
    main()
