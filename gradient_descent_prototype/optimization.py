import time
import math
from math import sqrt, atan
from itertools import chain

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.optimize import leastsq

from cell import Bacilli
from colony import LineageFrames
from lineage_funcs import load_colony

import dask.distributed

FONT = ImageFont.load_default()

debugcount = 0
badcount = 0  # DEBUG
is_cell = True
is_background = False


def objective(realimage, synthimage, cellmap, overlap_cost, cell_importance):
    """Full objective function between two images."""
    overlap_map = cellmap[cellmap > 1] - 1
    return np.sum(np.square((realimage - synthimage))) \
        + overlap_cost * np.sum(np.square(overlap_map))


def dist_objective(realimage, synthimage, distmap, cellmap, overlap_cost):
    overlap_map = cellmap[cellmap > 1] - 1
    return np.sum(np.square((realimage - synthimage) * distmap)) + overlap_cost * np.sum(np.square(overlap_map))


def find_optimal_simulation_conf(simulation_config, realimage1, cellnodes):
    shape = realimage1.shape

    def cost(values, target, simulation_config):
        for i in range(len(target)):
            simulation_config[target[i]] = values[i]
        synthimage, cellmap = generate_synthetic_image(cellnodes, shape, simulation_config)
        return (realimage1 - synthimage).flatten()

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

        for i, param in enumerate(variables):
            simulation_config[param] = optimal_values[i]
        simulation_config["cell.opacity"] = max(0, simulation_config["cell.opacity"])
        simulation_config["light.diffraction.sigma"] = max(0, simulation_config["light.diffraction.sigma"])

        if auto_opacity:
            for node in cellnodes:
                node.cell.opacity = simulation_config["cell.opacity"]

    print(simulation_config)
    return simulation_config


def generate_synthetic_image(cellnodes, shape, simulation_config):
    # image_type = simulation_config["image.type"]
    image_type = "graySynthetic"
    # print(image_type)
    cellmap = np.zeros(shape, dtype=int)
    if image_type == "graySynthetic" or image_type == "phaseContrast":
        
        background_color = simulation_config["background.color"]
        synthimage = np.full(shape, background_color)
        for node in cellnodes:
            node.cell.draw(synthimage, cellmap, is_cell, simulation_config)
        return synthimage, cellmap
    else:
        # print("I am here")
        synthimage = np.zeros(shape)
        for node in cellnodes:
            node.cell.draw(synthimage, cellmap, is_cell, simulation_config)
        return synthimage, cellmap


def load_image(imagefile):
    """Open the image file and convert to a floating-point grayscale array."""
    with open(imagefile, 'rb') as fp:
        realimage = np.array(Image.open(fp))
    if realimage.dtype == np.uint8:
        realimage = realimage.astype(float) / 255
    if len(realimage.shape) == 3:
        realimage = np.mean(realimage, axis=-1)
    return realimage


def perturb_bacilli(node, config, imageshape, invalid_limit=50):
    """Create a new perturbed bacilli cell."""
    global badcount  # DEBUG
    cell = node.cell
    prior = node.prior.cell

    if node.split:
        p1, p2 = node.prior.cell.split(node.alpha)
        if p1.name == node.cell.name:
            prior = p1
        elif p2.name == node.cell.name:
            prior = p2
        else:
            AssertionError('Names not matching')

    max_displacement = config['bacilli.maxSpeed'] / config['global.framesPerSecond']
    max_rotation = config['bacilli.maxSpin'] / config['global.framesPerSecond']
    min_growth = config['bacilli.minGrowth']
    max_growth = config['bacilli.maxGrowth']
    min_width = config['bacilli.minWidth']
    max_width = config['bacilli.maxWidth']
    min_length = config['bacilli.minLength']
    max_length = config['bacilli.maxLength']

    perturb_conf = config["perturbation"]
    p_x = perturb_conf["prob.x"]
    p_y = perturb_conf["prob.y"]
    p_width = perturb_conf["prob.width"]
    p_length = perturb_conf["prob.length"]
    p_rotation = perturb_conf["prob.rotation"]

    x_mu = perturb_conf["modification.x.mu"]
    y_mu = perturb_conf["modification.y.mu"]
    width_mu = perturb_conf["modification.width.mu"]
    length_mu = perturb_conf["modification.length.mu"]
    rotation_mu = perturb_conf["modification.rotation.mu"]

    x_sigma = perturb_conf["modification.x.sigma"]
    y_sigma = perturb_conf["modification.y.sigma"]
    width_sigma = perturb_conf["modification.width.sigma"]
    length_sigma = perturb_conf["modification.length.sigma"]
    rotation_sigma = perturb_conf["modification.rotation.sigma"]

    invalid_count = 0
    simulation_config = config["simulation"]
    if simulation_config["image.type"] == "graySynthetic":
        p_opacity = perturb_conf["prob.opacity"]

    while invalid_count < invalid_limit:
        # set starting properties
        x = cell.x
        y = cell.y
        width = cell.width
        length = cell.length
        rotation = cell.rotation
        cell_opacity = cell.opacity

        if simulation_config["image.type"] == "graySynthetic":
            p_decision = np.array([p_x, p_y, p_width, p_length, p_rotation, p_opacity])
        else:
            p_decision = np.array([p_x, p_y, p_width, p_length, p_rotation])

        p = np.random.uniform(0.0, 1.0, size=p_decision.size)

        # generate a sequence such that at least an attribute must be modified
        while (p > p_decision).all():
            p = np.random.uniform(0.0, 1.0, size=p_decision.size)

        if p[0] < p_decision[0]:  # perturb x
            x = cell.x + np.random.normal(x_mu, x_sigma)

        if p[1] < p_decision[1]:  # perturb y
            y = cell.y + np.random.normal(y_mu, y_sigma)

        if p[2] < p_decision[2]:  # perturb width
            width = cell.width + np.random.normal(width_mu, width_sigma)

        if p[3] < p_decision[3]:  # perturb length
            length = cell.length + np.random.normal(length_mu, length_sigma)

        if p[4] < p_decision[4]:  # perturb rotation
            rotation = cell.rotation + np.random.normal(rotation_mu, rotation_sigma)
        # if simulation_config["image.type"] == "graySynthetic" and p[5] < p_decision[5]:
            # cell_opacity = cell.opacity + (np.random.normal(opacity_mu, opacity_sigma))

        displacement = sqrt(np.sum((np.array([x, y, 0] - prior.position))**2))

        if not (0 <= x < imageshape[1] and 0 <= y < imageshape[0]) or (displacement > max_displacement) \
           or width < min_width or width > max_width or (abs(rotation - prior.rotation) > max_rotation) or \
           not (min_length < length < max_length) or not (min_growth < length - prior.length < max_growth):
            badcount += 1
            invalid_count += 1
        elif simulation_config["image.type"] == "graySynthetic" and cell_opacity < 0:
            badcount += 1
            invalid_count += 1
        else:
            break

    # push the new cell over the previous in the node
    node.push(Bacilli(cell.name, x, y, width, length, rotation, cell.split_alpha, cell_opacity))


def split_proba(length):
    """Returns the split probability given the length of the cell."""
    # Determined empirically based on previous runs
    return min(1, math.sin((length - 14) / (2 * math.pi * math.pi))) if 14 < length else 0


def bacilli_split(node, config, imageshape):
    """Split the cell and push both onto the stack for testing."""

    if np.random.random_sample() > split_proba(node.cell.length):
        return False

    max_displacement = config['bacilli.maxSpeed'] / config['global.framesPerSecond']
    max_rotation = config['bacilli.maxSpin'] / config['global.framesPerSecond']
    min_width = config['bacilli.minWidth']
    max_width = config['bacilli.maxWidth']
    min_length = config['bacilli.minLength']
    max_length = config['bacilli.maxLength']

    alpha = np.random.random_sample() / 5 + 2/5     # TODO choose from config
    cell1, cell2 = node.cell.split(alpha)

    # make sure that the lengths are within constraints
    if not (min_length < cell1.length < max_length and
            min_length < cell2.length < max_length):
        return False

    # split the prior to compare with for ensuring constraints are met
    pcell1, pcell2 = node.prior.cell.split(alpha)

    displacement = sqrt(np.sum((cell1.position - pcell1.position)**2))
    if not (0 <= cell1.position.x < imageshape[1] and
            0 <= cell1.position.y < imageshape[0]):
        return False
    elif displacement > max_displacement:
        return False
    elif not (min_width < cell1.width < max_width):
        return False
    elif abs(cell1.rotation - pcell1.rotation) > max_rotation:
        return False
    elif not (min_length < cell1.length < max_length):
        return False

    displacement = sqrt(np.sum((cell2.position - pcell2.position)**2))
    if not (0 <= cell2.position.x < imageshape[1] and
            0 <= cell2.position.y < imageshape[0]):
        return False
    elif displacement > max_displacement:
        return False
    elif not (min_width < cell2.width < max_width):
        return False
    elif abs(cell2.rotation - pcell2.rotation) > max_rotation:
        return False
    elif not (min_length < cell2.length < max_length):
        return False

    # push the split to the top of the cell stack
    node.parent.pop()
    node.parent.push2(cell1, cell2, alpha)

    return True


def bacilli_combine(node, config, imageshape):
    """Split the cell and push both onto the stack for testing."""
    if np.random.random_sample() > 0.2:
        return False, None

    max_displacement = config['bacilli.maxSpeed'] / config['global.framesPerSecond']
    max_rotation = config['bacilli.maxSpin'] / config['global.framesPerSecond']
    min_width = config['bacilli.minWidth']
    max_width = config['bacilli.maxWidth']
    min_length = config['bacilli.minLength']
    max_length = config['bacilli.maxLength']

    # get the cell node right before the split
    presplit = node.parent
    while len(presplit.children) < 2:
        presplit = presplit.parent

    # get the latest cell nodes after the split
    top_node1, top_node2 = presplit.children
    while top_node1.children:
        top_node1 = top_node1.children[0]
    while top_node2.children:
        top_node2 = top_node2.children[0]

    # combine the cells
    new_cell = top_node1.cell.combine(top_node2.cell)

    # compare with the prior for constraint checking
    prior_cell = presplit.prior.cell
    displacement = sqrt(np.sum((new_cell.position - new_cell.position)**2))
    if not (0 <= new_cell.position.x < imageshape[1] and
            0 <= new_cell.position.y < imageshape[0]):
        return False, None
    elif displacement > max_displacement:
        return False, None
    elif not (min_width < new_cell.width < max_width):
        return False, None
    elif abs(new_cell.rotation - prior_cell.rotation) > max_rotation:
        return False, None
    elif not (min_length < new_cell.length < max_length):
        return False, None

    # HACK
    # The following is somewhat of a work-around needed because the combined
    # cell cannot be pushed on top of split cells; therefore the previous
    # split cells will actually be on top instead of below the combined.

    # Here, push the new combined cell on top of the cell before the split;
    # then push the original two cells on top of the combined cell. The
    # stack will be treated differently after this point to account for this.
    presplit.pop()
    presplit.push(new_cell)
    new_node = presplit.children[0]
    if top_node1.cell.name == node.cell.name:
        new_node.push2(node.parent.cell, top_node2.cell, node.alpha)
        # new_node.children[0].push(top_node1.cell)
    elif top_node2.cell.name == node.cell.name:
        new_node.push2(top_node1.cell, node.parent.cell, node.alpha)
        # new_node.children[1].push(top_node2.cell)

    return True, presplit


# functions for different types of dynamic auto-temperature scheduling
def auto_temp_schedule_frame(frame, k_frame):
    return frame % k_frame == 0


def auto_temp_schedule_factor(cell_num, prev_num, factor):
    return True if cell_num / prev_num >= factor else False


def auto_temp_schedule_const(cell_num, prev_num, constant):
    return True if cell_num - prev_num >= constant else False


def auto_temp_schedule_cost(cost_diff):
    return True if ((cost_diff[1] - cost_diff[0]) / cost_diff[0]) > 0.2 else False


def optimize_core(imagefile, colony, args, config, iterations_per_cell=3000, auto_temp_complete=True, auto_const_temp = 1):
    """Core of the optimization routine."""
    global debugcount, badcount  # DEBUG

    bad_count = 0
    bad_prob_tot = 0

    realimage = load_image(imagefile)
    shape = realimage.shape
    simulation_config = config["simulation"]

    celltype = config['global.cellType'].lower()
    useDistanceObjective = args.dist

    cellnodes = list(colony)

    # find the initial cost
    synthimage, cellmap = generate_synthetic_image(cellnodes, shape, simulation_config)
    if useDistanceObjective:
        distmap = distance_transform_edt(realimage < .5)
        distmap /= config[f'{celltype}.distanceCostDivisor'] * config['global.pixelsPerMicron']
        distmap += 1
        cost = dist_objective(realimage, synthimage, distmap, cellmap, config["overlap.cost"])
    else:
        cost = objective(realimage, synthimage, cellmap, config["overlap.cost"], config["cell.importance"])

    # setup temperature schedule
    run_count = int(iterations_per_cell * len(cellnodes))

    if (auto_temp_complete == False):
        temperature = auto_const_temp
    else:
        temperature = args.start_temp
        end_temperature = args.end_temp

        alpha = (end_temperature / temperature)**(1 / run_count)

    for i in range(run_count):
        # print progress for debugging purposes
        #if i%1013 == 59:
        #    print(f'{imagefile.name}: Progress: {100*i/run_count:.02f}%', flush=True)

        # choose a cell at random
        index = np.random.randint(0, len(cellnodes))
        node = cellnodes[index]

        # perturb the cell and push it onto the stack
        if celltype == 'bacilli':
            perturb_bacilli(node, config, shape)
            new_node = node.children[0]

            old_synthimage = synthimage.copy()
            old_cellmap = cellmap.copy()

            # # try splitting (or combining if already split)
            combined = False
            split = False
            if node.split:
                combined, presplit = bacilli_combine(new_node, config, shape)
            elif not node.split:
                split = bacilli_split(new_node, config, shape)

            if combined:
                # get the new combined node
                cnode = presplit.children[0]

                # get the previous split nodes (see note in bacilli_combine)
                snode1, snode2 = cnode.children

                # compute the starting cost
                region = (cnode.cell.simulated_region(simulation_config).
                          union(snode1.cell.simulated_region(simulation_config))
                          .union(snode2.cell.simulated_region(simulation_config)))
                if useDistanceObjective:
                    start_cost = dist_objective(
                        realimage[region.top:region.bottom, region.left:region.right],
                        synthimage[region.top:region.bottom, region.left:region.right],
                        distmap[region.top:region.bottom, region.left:region.right],
                        cellmap[region.top:region.bottom, region.left:region.right],
                        config["overlap.cost"])
                else:
                    start_cost = objective(realimage[region.top:region.bottom,region.left:region.right],
                                           synthimage[region.top:region.bottom,region.left:region.right],
                                           cellmap[region.top:region.bottom,region.left:region.right],
                                           config["overlap.cost"],
                                           config["cell.importance"])

                # subtract the previous cells
                snode1.cell.draw(synthimage, cellmap, is_background, simulation_config)
                snode2.cell.draw(synthimage, cellmap, is_background, simulation_config)

                # add the new cell
                cnode.cell.draw(synthimage, cellmap, is_cell, simulation_config)

            elif split:
                snode1, snode2 = node.children

                # compute the starting cost
                region = (node.cell.simulated_region(simulation_config).\
                          union(snode1.cell.simulated_region(simulation_config))\
                          .union(snode2.cell.simulated_region(simulation_config)))
                if useDistanceObjective:
                    start_cost = dist_objective(
                        realimage[region.top:region.bottom, region.left:region.right],
                        synthimage[region.top:region.bottom, region.left:region.right],
                        distmap[region.top:region.bottom, region.left:region.right],
                        cellmap[region.top:region.bottom, region.left:region.right],
                        config["overlap.cost"])
                else:
                    start_cost = objective(realimage[region.top:region.bottom,region.left:region.right],
                                           synthimage[region.top:region.bottom,region.left:region.right],
                                           cellmap[region.top:region.bottom,region.left:region.right],
                                           config["overlap.cost"],
                                           config["cell.importance"])

                # subtract the previous cell
                node.cell.draw(synthimage, cellmap, is_background, simulation_config)

                # add the new cells
                snode1.cell.draw(synthimage, cellmap, is_cell, simulation_config)
                snode2.cell.draw(synthimage, cellmap, is_cell, simulation_config)

            else:
                # compute the starting cost
                region = node.cell.simulated_region(simulation_config).\
                union(new_node.cell.simulated_region(simulation_config))
                if useDistanceObjective:
                    start_cost = dist_objective(
                        realimage[region.top:region.bottom, region.left:region.right],
                        synthimage[region.top:region.bottom, region.left:region.right],
                        distmap[region.top:region.bottom, region.left:region.right],
                        cellmap[region.top:region.bottom, region.left:region.right],
                        config["overlap.cost"])
                else:
                    start_cost = objective(realimage[region.top:region.bottom,region.left:region.right],
                                           synthimage[region.top:region.bottom,region.left:region.right],
                                           cellmap[region.top:region.bottom,region.left:region.right],
                                           config["overlap.cost"],
                                           config["cell.importance"])

                # subtract the previous cell
                node.cell.draw(synthimage, cellmap, is_background, simulation_config)

                # add the new cells
                new_node.cell.draw(synthimage, cellmap, is_cell, simulation_config)
            
            if useDistanceObjective:
                    end_cost = dist_objective(
                        realimage[region.top:region.bottom, region.left:region.right],
                        synthimage[region.top:region.bottom, region.left:region.right],
                        distmap[region.top:region.bottom, region.left:region.right],
                        cellmap[region.top:region.bottom, region.left:region.right],
                        config["overlap.cost"])
            else:
                end_cost = objective(realimage[region.top:region.bottom,region.left:region.right],
                                     synthimage[region.top:region.bottom,region.left:region.right],
                                     cellmap[region.top:region.bottom,region.left:region.right],
                                     config["overlap.cost"],
                                     config["cell.importance"])
            costdiff = end_cost - start_cost

            # compute the acceptance threshold
            if costdiff <= 0:
                acceptance = 1.0
            else:
                acceptance = np.exp(-costdiff / temperature)
                bad_count += 1
                bad_prob_tot += acceptance

            # check if the acceptance threshold was met; pop if not
            if acceptance <= np.random.random_sample():
                # restore the previous cells
                if combined:
                    presplit.pop()
                    presplit.push2(snode1.cell, snode2.cell, node.alpha)
                else:
                    node.pop()

                # restore the diff image
                synthimage = old_synthimage
                cellmap = old_cellmap

            else:
                if combined:
                    presplit.children[0].pop()

                cost += costdiff

            colony.flatten()
            cellnodes = list(colony)

            # DEBUG
            if args.debug and i % 80 == 0:
                synthimage, cellmap = generate_synthetic_image(cellnodes, shape, simulation_config)
                frame_1 = np.empty((shape[0], shape[1], 3))
                frame_1[..., 0] = (realimage - synthimage)
                frame_1[..., 1] = frame_1[..., 0]
                frame_1[..., 2] = frame_1[..., 0]

                # for node in cellnodes:
                    # node.cell.drawoutline(frame_1, (1, 0, 0))

                frame_1 = np.clip(frame_1, 0, 1)

                debugimage = Image.fromarray((255 * frame_1).astype(np.uint8))
                debugimage.save(args.debug / f'residual{debugcount}.png')

                frame_2 = np.empty((shape[0], shape[1], 3))
                frame_2[..., 0] = synthimage
                frame_2[..., 1] = frame_2[..., 0]
                frame_2[..., 2] = frame_2[..., 0]

                # for node in cellnodes:
                    #.node.cell.drawoutline(frame_2, (1, 0, 0))

                frame_2 = np.clip(frame_2, 0, 1)

                debugimage = Image.fromarray((255 * frame_2).astype(np.uint8))
                debugimage.save(args.debug / f'synthetic{debugcount}.png')
                debugcount += 1

        if auto_temp_complete:
            temperature *= alpha

    # print(f'Bad Percentage: {100*badcount/run_count}%')

    if not auto_temp_complete:

        print("pbad is ", bad_prob_tot / bad_count)
        # print("temperature is ", temperature)

        return bad_prob_tot / bad_count

    if useDistanceObjective:
        new_cost = dist_objective(realimage, synthimage, distmap, cellmap, config["overlap.cost"])
    else:
        new_cost = objective(realimage, synthimage, cellmap, config["overlap.cost"], config["cell.importance"])

    colony.set_cost(cost)

    print(f'Incremental Cost: {cost}')
    print(f'Actual Cost:      {new_cost}')
    if abs(new_cost - cost) > 1e-7:
        print('WARNING: incremental cost diverged from expected cost')

    frame = np.empty((shape[0], shape[1], 3))
    frame[..., 0] = realimage
    frame[..., 1] = frame[..., 0]
    frame[..., 2] = frame[..., 0]

    for node in cellnodes:
        node.cell.drawoutline(frame, (1, 0, 0))

    frame = np.clip(frame, 0, 1)

    debugimage = Image.fromarray((255 * frame).astype(np.uint8))

    best_fit_frame = np.empty((shape[0], shape[1], 3))
    best_fit_frame[..., 0] = synthimage
    best_fit_frame[..., 1] = best_fit_frame[..., 0]
    best_fit_frame[..., 2] = best_fit_frame[..., 0]
    best_fit_frame = np.clip(best_fit_frame, 0, 1)
    best_fit_frame = Image.fromarray((255 * best_fit_frame). astype(np.uint8))
    return colony, cost, debugimage, best_fit_frame


def auto_temp_schedule(imagefile, colony, args, config):
    initial_temp = 1
    ITERATION = 500
    AUTO_TEMP_COMPLETE = False

    count = 0

    while(optimize_core(imagefile, colony, args, config, ITERATION, AUTO_TEMP_COMPLETE, initial_temp) < 0.40):
        count += 1
        initial_temp *= 10.0
        print(f"count: {count}")
    print("finished < 0.4")
    while(optimize_core(imagefile, colony, args, config, ITERATION, AUTO_TEMP_COMPLETE, initial_temp) > 0.40):
        count += 1 
        initial_temp /= 10.0
        print(f"count: {count}")
    print("finished > 0.4")
    while(optimize_core(imagefile, colony, args, config, ITERATION, AUTO_TEMP_COMPLETE, initial_temp) < 0.40):
        count += 1
        initial_temp *= 1.1
        print(f"count: {count}")
    end_temp = initial_temp
    print("finished < 0.4")
    while(optimize_core(imagefile, colony, args, config, ITERATION, AUTO_TEMP_COMPLETE, end_temp) >= 1e-10):
        count += 1
        end_temp /= 10.0

    return initial_temp, end_temp


def optimize(imagefile, lineageframes, args, config, client):
    """Optimize the cell properties using simulated annealing."""
    global badcount  # DEBUG
    badcount = 0  # DEBUG

    if not client:
        colony, _, debugimage, best_fit_frame = optimize_core(imagefile, lineageframes.forward(), args, config)
        debugimage.save(args.output / imagefile.name)
        best_fit_frame.save(args.bestfit / imagefile.name)
        return colony

    # tasks = []

    group = lineageframes.latest_group
    ejob = args.jobs // len(group)

    futures = []

    for colony in group:
        for i in range(ejob):
            newColony = colony.clone()
            futures.append(client.submit(optimize_core, imagefile, newColony, args, config))

    try:
        dask.distributed.wait(futures, 360)
    except Exception as e:
        print(e)

    results = []
    for future in futures:
        if not future.done():
            print('Task timed out - Cancelling')
            future.cancel()
        else:
            results.append(future.result(timeout=10))

    if args.strategy not in ['best-wins', 'worst-wins', 'extreme-wins']:
        raise ValueError('--strategy must be one of "best-wins", "worst-wins", "extreme-wins"')

    if args.strategy in ['best-wins', 'worst-wins']:
        results = sorted(results, key=lambda x: x[1], reverse=args.strategy == 'worst-wins')
    else:
        bestresults = sorted(results, key=lambda x: x[1])
        worstresults = sorted(results, key=lambda x: x[1], reverse=True)
        # https://stackoverflow.com/a/11125256
        results = list(chain.from_iterable(zip(bestresults, worstresults)))

    winning = results[:args.keep]
    print('keeping {}, got {}'.format(args.keep, len(winning)))

    # Choose the best or worst
    print('The winning solution(s) ({}) have cost values {}'.format(args.strategy, [s[1] for s in winning]))
    print('CHECKPOINT, {}, {}, {}'.format(time.time(), imagefile.name, winning[0][1]), flush=True)

    lineageframes.add_frame([s[0] for s in winning])

    for i, s in enumerate(winning):
        debugimage = s[2]
        debugimage.save(args.output / '{:03d}-{}'.format(i, imagefile.name))

    return colony


def update_cost_diff(colony, cost_diff):
    if -1 == cost_diff[0]:
        cost_diff = (colony.cost, cost_diff[1])
    elif -1 == cost_diff[1]:
        cost_diff = (cost_diff[0], colony.cost)
    else:
        cost_diff = (cost_diff[1], colony.cost)

    return cost_diff


def local_optimize(imagefiles, config, args, lineagefile, client):
    lineageframes = LineageFrames()
    colony = lineageframes.forward()
    if args.lineage_file:
        load_colony(colony, args.lineage_file, config, initial_frame=imagefiles[0].name)
    else:
        load_colony(colony, args.initial, config)
    cost_diff = (-1, -1)
    celltype = config['global.cellType'].lower()

    config["simulation"] = find_optimal_simulation_conf(config["simulation"], load_image(imagefiles[0]), list(colony))
    if args.auto_temp == 1:
        print("auto temperature schedule started")
        args.start_temp, args.end_temp = auto_temp_schedule(imagefiles[0], lineageframes.forward(), args, config)
        print("auto temperature schedule finished")
        print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)

    frame_num = 0
    prev_cell_num = len(colony)
    for imagefile in imagefiles:  # Recomputing temperature when needed

        frame_num += 1

        if args.auto_meth == "frame":
            if auto_temp_schedule_frame(frame_num, 8):
                print("auto temperature schedule started (recomputed)")
                args.start_temp, args.end_temp = auto_temp_schedule(imagefile, colony, args, config)
                print("auto temperature schedule finished")
                print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)

        elif args.auto_meth == "factor":
            if auto_temp_schedule_factor(len(colony), prev_cell_num, 1.1):
                print("auto temperature schedule started (recomputed)")
                args.start_temp, args.end_temp = auto_temp_schedule(imagefile, colony, args, config)
                print("auto temperature schedule finished")
                print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                prev_cell_num = len(colony)

        elif args.auto_meth == "const":
            if auto_temp_schedule_const(len(colony), prev_cell_num, 10):
                print("auto temperature schedule started (recomputed)")
                args.start_temp, args.end_temp = auto_temp_schedule(imagefile, colony, args, config)
                print("auto temperature schedule finished")
                print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)
                prev_cell_num = len(colony)

        elif args.auto_meth == "cost":
            print(cost_diff, frame_num, auto_temp_schedule_cost(cost_diff))
            if frame_num >= 2 and auto_temp_schedule_cost(cost_diff):
                print("auto temperature schedule started cost_diff (recomputed)")
                args.start_temp, args.end_temp = auto_temp_schedule(imagefile, colony, args, config)
                print("auto temperature schedule finished")
                print("starting temperature is ", args.start_temp, "ending temperature is ", args.end_temp)

        colony = optimize(imagefile, lineageframes, args, config, client)

        cost_diff = update_cost_diff(colony, cost_diff)

        colony.flatten()

        for cellnode in colony:
            properties = [imagefile.name, cellnode.cell.name]
            if celltype == 'bacilli':
                properties.extend([
                    str(cellnode.cell.x),
                    str(cellnode.cell.y),
                    str(cellnode.cell.width),
                    str(cellnode.cell.length),
                    str(cellnode.cell.rotation)])
            print(','.join(properties), file=lineagefile)
