from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.optimize import leastsq
from PIL import Image

from .FrameM import FrameM
import numpy as np
import pandas as pd
from .helper import get_input_file_paths, load_image

from ..Cells.Sphere import Sphere

SCALING = 6

class LineageM:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        simulation_config = config["simulation"]
        # self.real_images = real_images

        if 'image.slices' in self.config["simulation"]:
            self.z_slices = list(map(lambda x: SCALING * x, (range(-1 * (self.config["simulation"]['image.slices'] // 2), self.config["simulation"]['image.slices'] // 2 + 1))))
        else:
            self.z_slices = [0]

        self.load_real_images()

        if 'padding' in simulation_config:
            self.pad = simulation_config["padding"]
        else:
            self.pad = 0

        self.pad_real_image(0)
        self.frames = [FrameM(simulation_config)]
        self.synth_image_stacks = []
        self.cell_map_stacks = []
        self.dist_map_stacks = []
        self.real_images = self.load_real_images()

        shape = self.real_images[0][0].shape
        self.shape = (shape[0] + 2 * self.pad, shape[1] + 2 * self.pad)

        continue_from = args.continue_from
        lineage_file = args.lineage_file or args.initial
        cells_data = pd.read_csv(lineage_file)
        cells_data = cells_data.replace('None', None)
        for i in range(len(self.file_paths)):
            file_name = self.file_paths[i][0][0].name
            current_frame_number = self.file_paths[i][0][1]
            if current_frame_number > continue_from:
                break
            if i > 0:
                self.forward()
            for _, row in cells_data[cells_data["file"] == file_name].iterrows():
                new_cell = Sphere(row["name"], row["x"], row["y"], 2 * row["r"], row["split_alpha"], row["opacity"], z = SCALING * (row["z"] - 16))
                self.frames[i].add_cell(new_cell)
            self.frames[i].simulation_config = self.find_optimal_simulation_config(i)
            synth_image_stack, cell_map_stack = self.generate_synthetic_images(i)
            self.synth_image_stacks.append(synth_image_stack)
            self.cell_map_stacks.append(cell_map_stack)

            for i in range(len(self.frames)):
                for cellnode in self.frames[i].node_map.values():
                    cellnode.cell.x = cellnode.cell.x + self.pad
                    cellnode.cell.y = cellnode.cell.y + self.pad

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
        self.pad_real_image(len(self.frames) - 1)

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

    def load_real_images(self):
        self.file_paths = get_input_file_paths(self.args, len(self.z_slices))
        return [[load_image(file_path[0]) for file_path in file_path_stack] for file_path_stack in self.file_paths]

    def pad_real_image(self, frame_index):
        if (self.pad > 0):
            self.real_images[frame_index] = np.pad(self.real_images[frame_index], ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant', constant_values=0)

    def generate_synthetic_images(self, frame_index):
        # Generates a stack of synthetic images for the given frame index
        if frame_index > len(self.frames):
            raise RuntimeError("frame_index is out of range")
        image_type = self.config["simulation"]["image.type"]
        shape = self.shape
        cell_nodes = self.frames[frame_index].nodes
        simulation_config = self.frames[frame_index].simulation_config

        synth_image_stack = []
        cell_map_stack = []
        if image_type == "graySynthetic" or image_type == "phaseContrast":
            background_color = self.config["simulation"]["background.color"]
        else:
            background_color = 0

        for z in self.z_slices:
            synth_image = np.full(shape, background_color)
            cell_map = np.zeros(shape, dtype=int)
            for node in cell_nodes:
                node.cell.draw(synth_image, cell_map, True, simulation_config, z = z)
            synth_image_stack.append(synth_image)
            cell_map_stack.append(cell_map)

        if frame_index == len(self.synth_image_stacks):
            self.synth_image_stacks.append(synth_image_stack)
            self.cell_map_stacks.append(cell_map_stack)
        else:
            self.synth_image_stacks[frame_index] = synth_image_stack
            self.cell_map_stacks[frame_index] = cell_map_stack
        return synth_image_stack, cell_map_stack

    def find_optimal_simulation_config(self, frame_index):
        real_image_stack = np.array(self.real_images[frame_index])
        cell_nodes = self.frames[frame_index].nodes
        simulation_config = self.frames[frame_index].simulation_config

        def cost(values, target, simulation_config):
            for i in range(len(target)):
                simulation_config[target[i]] = values[i]
            synth_image_stack, _ = self.generate_synthetic_images(frame_index)
            synth_image_stack = np.array(synth_image_stack)
            return (real_image_stack - synth_image_stack).flatten()

        initial_values = []
        variables = []
        if simulation_config["background.color"] == "auto":
            variables.append("background.color")
            initial_values.append(1)
        if simulation_config["cell.color"] == "auto":
            variables.append("cell.color")
            initial_values.append(0)
        if simulation_config["light.diffraction.sigma"] == "auto":
            variables.append("light.diffraction.sigma")
            initial_values.append(11)
        if simulation_config["light.diffraction.strength"] == "auto":
            variables.append("light.diffraction.strength")
            initial_values.append(0.5)
        if simulation_config["cell.opacity"] == "auto":
            auto_opacity = True
            variables.append("cell.opacity")
            initial_values.append(0.2)
        if len(variables) != 0:
            residues = lambda x: cost(x, variables, simulation_config)
            optimal_values, _ = leastsq(residues, initial_values)

            for frame_index, param in enumerate(variables):
                simulation_config[param] = optimal_values[frame_index]
            simulation_config["cell.opacity"] = max(0, simulation_config["cell.opacity"])
            simulation_config["light.diffraction.sigma"] = max(0, simulation_config["light.diffraction.sigma"])

            if auto_opacity:
                for node in cell_nodes:
                    node.cell.opacity = simulation_config["cell.opacity"]

        print(f"optimal simulation configuration values found: {simulation_config}")
        return simulation_config

    def save_output(self, frame_index):
        real_image_stack = self.real_images[frame_index]
        synth_image_stack = self.synth_image_stacks[frame_index]

        config = self.config

        residual_vmin = config["residual.vmin"]
        residual_vmax = config["residual.vmax"]
        if self.args.residual:
            colormap = cm.ScalarMappable(norm=Normalize(vmin=residual_vmin, vmax=residual_vmax), cmap="bwr")

        for i, z in enumerate(self.z_slices):
            synth_image = Image.fromarray(np.uint8(255 * synth_image_stack[i]), "L")
            synth_image.save(self.args.bestfit / self.file_paths[frame_index][i][0].name)
            output_frame = np.stack((real_image_stack[i],) * 3, axis=-1)
            for node in self.frames[frame_index].nodes:
                if node.cell.dormant:
                    continue
                node.cell.drawoutline(output_frame, (1, 0, 0), z)
            output_frame = Image.fromarray(np.uint8(255 * output_frame))
            output_frame.save(self.args.output / self.file_paths[frame_index][i][0].name)
            print(f"saved {self.file_paths[frame_index][i][0].name} (z = {z})")

        if self.args.residual:
            for i, z in enumerate(self.z_slices):
                residual = real_image_stack[i] - synth_image_stack[i]
                residual = np.clip(residual, residual_vmin, residual_vmax)
                residual = colormap.to_rgba(residual)
                residual = Image.fromarray(np.uint8(255 * residual))
                residual.save(self.args.residual / self.file_paths[frame_index][z][0].name)

    def save_to_file(self, frame_index, lineagefile):
        for node in self.frames[frame_index].nodes:
            properties = [self.file_paths[frame_index][0][0].name, node.cell.name]
            properties.extend([
                str(node.cell.x),
                str(node.cell.y),
                str(node.cell.width),
                str(node.cell.length),
                str(node.cell.rotation),
                str(node.cell.split_alpha),
                str(node.cell.opacity)])
            print(','.join(properties), file=lineagefile)

    def get_frame_stacks_at_index(self, frame_index):
        return self.synth_image_stacks[frame_index], self.cell_map_stacks[frame_index], [], self.real_images[frame_index]

