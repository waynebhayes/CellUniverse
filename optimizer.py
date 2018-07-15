# -*- coding: utf-8 -*-

"""
cellannealer.optimizer
~~~~~~~~~~~~~~~~~~~~~~
"""

import logging
import random
from copy import deepcopy
from math import sqrt
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from colony import Colony, LineageNode

logger = logging.getLogger(__name__)

DEBUG = True
DEBUG_COUNT = 0
DEBUG_INTERVAL = 80

FONT_SIZE = 16
FONT = ImageFont.truetype('Hack-Regular.ttf', size=FONT_SIZE)


def objective(real_image: np.ndarray, synthetic_image: np.ndarray) -> float:
    difference = real_image - synthetic_image
    return np.sum(np.power(difference, 2))


def simulated_annealing(colony: Colony, real_image: np.ndarray, config: dict,
                        filename: Path, offset: int):

    global DEBUG_COUNT

    # config constraints
    max_displacement = config['maxSpeed']*config['timestep']
    max_rotation = config['maxSpin']*config['timestep']
    min_width = config['minWidth']
    max_width = config['maxWidth']
    min_growth = config['minGrowth']
    max_growth = config['maxGrowth']
    min_length = config['minLength']
    max_length = config['maxLength']

    original_colony = deepcopy(colony)

    leaves = colony.leaves
    original_leaves = original_colony.leaves

    run_count = int(500*len(leaves))
    temperature = 5.0
    alpha = 10**(np.log10(0.99)/len(leaves))

    # ensure no cells start in flux
    for leaf in leaves:
        leaf.cell.in_flux = False

    # find initial cost
    synthetic_image = np.zeros_like(real_image) - 0.5
    for drawing_leaf in leaves:
        drawing_leaf.cell.draw(synthetic_image)
    cost = objective(real_image, synthetic_image)

    neg_accept = 0
    neg_reject = 0

    for i in range(run_count):
        if i%101 == 59:
            neg_accept_percentage = 100*neg_accept/(neg_accept + neg_reject)
            logger.info(f'Iteration: {i}  Temperature: {temperature:.03e}  Neg-Accept: {neg_accept_percentage:.02f}%')
            neg_accept = 0
            neg_reject = 0

        index = random.randint(0, len(leaves) - 1)

        leaf = leaves[index]
        original_leaf = original_leaves[index]

        # ensure existing is valid
        displacement = sqrt(np.sum((leaf.cell.position - original_leaf.cell.position)**2))
        rotation = abs(leaf.cell.rotation - original_leaf.cell.rotation)
        growth = leaf.cell.dimensions.length - original_leaf.cell.dimensions.length

        assert displacement <= max_displacement
        assert leaf.cell.dimensions.width >= min_width
        assert leaf.cell.dimensions.width <= max_width
        assert rotation <= max_rotation
        # assert growth >= min_growth and growth <= max_growth
        # assert growth <= max_growth
        assert leaf.cell.dimensions.length >= min_length
        assert leaf.cell.dimensions.length <= max_length

        new_position = leaf.cell.position.copy()
        new_rotation = leaf.cell.rotation
        new_dimensions = leaf.cell.dimensions.copy()

        no_change = True
        while no_change:
            choice = random.random()
            if choice < 0.35:
                new_position.x += random.gauss(0, sigma=.5)
                no_change = False

            choice = random.random()
            if choice < 0.35:
                new_position.y += random.gauss(0, sigma=.5)
                no_change = False

            choice = random.random()
            if choice < 0.2:
                new_rotation += random.gauss(0, sigma=0.1)
                no_change = False

            choice = random.random()
            if choice < 0.2:
                new_dimensions.length += random.gauss(0, sigma=1)
                no_change = False

            choice = random.random()
            if choice < 0.1:
                new_dimensions.width += random.gauss(0, sigma=0.05)
                no_change = False

        leaf.push()
        original_leaf.push()

        # Test new changes
        leaf.cell.update(
            position=new_position,
            rotation=new_rotation,
            dimensions=new_dimensions)

        combined = False
        skip = False

        choice = random.random()

        if choice < 0.05 and not leaf.cell.in_flux:     # split cell
            # choose alpha
            length_alpha = random.random()/5 + 2/5     # TODO: choose from settings and constraints
            # length_alpha = 0.5

            # split
            cell_1, cell_2 = leaf.cell.split(length_alpha)

            if (min_length < cell_1.dimensions.length < max_length and
                    min_length < cell_2.dimensions.length < max_length):

                leaf.children.append(LineageNode(cell_1, parent=leaf))
                leaf.children.append(LineageNode(cell_2, parent=leaf))

                # also split original
                original_cell_1, original_cell_2 = original_leaf.cell.split(length_alpha)
                original_leaf.children.append(LineageNode(original_cell_1, parent=original_leaf))
                original_leaf.children.append(LineageNode(original_cell_2, parent=original_leaf))

                # update leaves
                leaves = colony.leaves
                original_leaves = colony.leaves

                # Enforce constraints
                displacement = sqrt(np.sum((cell_1.position - original_cell_1.position)**2))
                rotation = abs(cell_1.rotation - original_cell_1.rotation)
                growth = cell_1.dimensions.length - original_cell_1.dimensions.length

                if displacement > max_displacement:
                    skip = True

                if cell_1.dimensions.width < min_width or cell_1.dimensions.width > max_width:
                    skip = True

                if rotation > max_rotation:
                    skip = True

                if growth < min_growth or growth > max_growth:
                    skip = True

                if cell_1.dimensions.length < min_length or cell_1.dimensions.length > max_length:
                    skip = True

                # Enforce constraints
                displacement = sqrt(np.sum((cell_2.position - original_cell_2.position)**2))
                rotation = abs(cell_2.rotation - original_cell_2.rotation)
                growth = cell_2.dimensions.length - original_cell_2.dimensions.length

                if displacement > max_displacement:
                    skip = True

                if cell_2.dimensions.width < min_width or cell_2.dimensions.width > max_width:
                    skip = True

                if rotation > max_rotation:
                    skip = True

                if growth < min_growth or growth > max_growth:
                    skip = True

                if cell_2.dimensions.length < min_length or cell_2.dimensions.length > max_length:
                    skip = True

        elif choice < 0.2 and leaf.cell.in_flux:    # combine cells
            combined = True

            # get the parent node
            parent = leaf.parent
            original_parent = original_leaf.parent

            # push parents
            parent.push()
            original_parent.push()

            # update the parent cell
            parent.cell.combine(parent.children[0].cell, parent.children[1].cell)
            original_parent.cell.combine(original_parent.children[0].cell, original_parent.children[1].cell)

            # remove the children
            parent.children.clear()
            original_parent.children.clear()

            # update leaves
            leaves = colony.leaves
            original_leaves = original_colony.leaves

            # Enforce constraints
            displacement = sqrt(np.sum((parent.cell.position - original_parent.cell.position)**2))
            rotation = abs(parent.cell.rotation - original_parent.cell.rotation)
            growth = parent.cell.dimensions.length - original_parent.cell.dimensions.length

            if displacement > max_displacement:
                skip = True

            if parent.cell.dimensions.width < min_width or parent.cell.dimensions.width > max_width:
                skip = True

            if rotation > max_rotation:
                skip = True

            if growth < min_growth or growth > max_growth:
                skip = True

            if parent.cell.dimensions.length < min_length or parent.cell.dimensions.length > max_length:
                skip = True

        # Enforce constraints
        displacement = sqrt(np.sum((new_position - original_leaf.cell.position)**2))
        rotation = abs(new_rotation - original_leaf.cell.rotation)
        growth = new_dimensions.length - original_leaf.cell.dimensions.length

        if displacement > max_displacement:
            skip = True

        if new_dimensions.width < min_width or new_dimensions.width > max_width:
            skip = True

        if rotation > max_rotation:
            skip = True

        if growth < min_growth or growth > max_growth:
            skip = True

        if new_dimensions.length < min_length or new_dimensions.length > max_length:
            skip = True

        if skip:
            leaf.pop()
            original_leaf.pop()
            if combined:
                parent.pop()
                original_parent.pop()
        else:
            synthetic_image = np.zeros_like(real_image)

            for drawing_leaf in leaves:
                drawing_leaf.cell.draw(synthetic_image)

            new_cost = objective(real_image, synthetic_image)
            change = new_cost - cost

            # Test if accepted
            if change <= 0:
                acceptance = 1.0
            else:
                acceptance = np.exp(-change/temperature)

            if acceptance <= random.random():

                neg_reject += 1

                leaf.pop()
                original_leaf.pop()
                if combined:
                    parent.pop()
                    original_parent.pop()
            else:

                if change > 0:
                    neg_accept += 1

                leaf.flatten()
                original_leaf.flatten()
                if combined:
                    parent.flatten()
                    original_parent.flatten()
                cost = new_cost

        # update leaves
        leaves = colony.leaves
        original_leaves = original_colony.leaves

        if DEBUG and i%DEBUG_INTERVAL == 0:
            # Produce output frame and save
            synthetic_image = np.zeros_like(real_image) - 0.5

            for drawing_leaf in leaves:
                drawing_leaf.cell.draw(synthetic_image)

            frame = np.empty((synthetic_image.shape[0], synthetic_image.shape[1], 3))
            frame[:, :, 0] = (real_image - synthetic_image + 1.0)/2
            frame[:, :, 1] = frame[:, :, 0]
            frame[:, :, 2] = frame[:, :, 0]

            for drawing_leaf in leaves:
                drawing_leaf.cell.debug_draw(frame)

            frame = np.clip(frame, 0, 1)

            debug_image = Image.fromarray((255*frame).astype(np.uint8))
            draw = ImageDraw.Draw(debug_image)
            draw.text((2, 2), str(filename), (0, 0, 0), FONT)

            debug_image.save(f'./debug/frame{DEBUG_COUNT:04}.png')
            DEBUG_COUNT += 1

        # Cool temperature
        temperature *= alpha

    # Produce output frame and save
    frame = np.empty((real_image.shape[0], real_image.shape[1], 3))
    frame[:, :, 0] = real_image/2
    frame[:, :, 1] = frame[:, :, 0]
    frame[:, :, 2] = frame[:, :, 0]

    for drawing_leaf in leaves:
        drawing_leaf.cell.debug_draw(frame)

    frame = np.clip(frame, 0, 1)

    output_image = Image.fromarray((255*frame).astype(np.uint8))
    draw = ImageDraw.Draw(output_image)
    draw.text((2, 2), str(filename), (255, 255, 0), FONT)

    output_image.save(f'./output/{filename.name}')
