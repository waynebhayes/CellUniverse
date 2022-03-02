from .FrameM import FrameM
import numpy as np


class LineageM:
    def __init__(self, simulation_config=None):
        self.frames = [FrameM(simulation_config)]

    def __repr__(self):
        return '\n'.join([str(frame) for frame in self.frames])

    @property
    def total_cell_count(self):
        return sum(len(frame.node_map) for frame in self.frames)

    def count_cells_in(self, start, end):
        if start is None or start < 0:
            start = 0
        if end is None or end > len(self.frames):
            end = len(self.frames)
        return sum(len(frame.node_map) for frame in self.frames[start:end])

    def forward(self):
        self.frames.append(FrameM(self.frames[-1].simulation_config, self.frames[-1]))

    def copy_forward(self):
        self.forward()
        for cell_node in self.frames[-2].nodes:
            self.frames[-1].add_cell(cell_node.cell)

    def choose_random_frame_index(self, start=None, end=None) -> int:
        if start is None or start < 0:
            start = 0

        if end is None or end > len(self.frames):
            end = len(self.frames)

        threshold = int(np.random.random_sample() * self.count_cells_in(start, end))

        for i in range(start, end):
            frame = self.frames[i]
            if len(frame.nodes) > threshold:
                return i
            else:
                threshold -= len(frame.nodes)

        raise RuntimeError('this should not have happened')
