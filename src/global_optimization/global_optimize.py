from copy import deepcopy
import numpy as np
import dask

import optimization
from .Changes import BackGroundLuminosityOffset, CameraShift, Combination, OpacityDiffractionOffset, Perturbation, Split
from .utils import gerp
from .Modules import CellNodeM, FrameM, LineageM

totalCostDiff = 0.0


def global_optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                    iteration_per_cell, client=None, in_auto_temp_schedule=False, const_temp=None):

    window = window_end - window_start
    total_iterations = iteration_per_cell * lineage.count_cells_in(window_start, window_end) // window
    batch_size = total_iterations // args.batches

    if in_auto_temp_schedule:
        lineage = deepcopy(lineage)
        synthimages = deepcopy(synthimages)
        distmaps = deepcopy(distmaps)
        cellmaps = deepcopy(cellmaps)

        return optimize_core(lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, args.start_temp, args.end_temp, config, iteration_per_cell, 0, total_iterations, total_iterations, in_auto_temp_schedule, const_temp)

    if client is None:
        # single threading
        cost, lineage, synthimages, distmaps, cellmaps = optimize_core(lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, args.start_temp, args.end_temp, config, iteration_per_cell, 0, total_iterations, total_iterations, in_auto_temp_schedule, const_temp)
        return lineage, synthimages, distmaps, cellmaps

    # TODO: Figure out how to get debugfile back without messing with dask
    # debugfile = None
    # if args.debug:
    #     debugfile = open(args.debug/'debug.csv', 'a')

    # slice data to reduce size when threading
    # makes current frame start at window 0 so offset is required
    temp_lineage = LineageM(config['simulation'])
    temp_lineage.frames = lineage.frames[window_start:window_end]
    temp_realimages = realimages[window_start:window_end]
    temp_synthimages = synthimages[window_start:window_end]
    temp_cellmaps = cellmaps[window_start:window_end]
    temp_distmaps = distmaps[window_start:window_end]
    current_iteration = 0

    while current_iteration < total_iterations:
        print(current_iteration, total_iterations)
        futures = []
        for i in range(args.jobs):
            # DO NOT USE client.scatter()
            # TODO: suppress warnings that advise scatter
            futures.append(client.submit(optimize_core, temp_lineage, temp_realimages, temp_synthimages, temp_cellmaps, temp_distmaps, window_start, window_end, args.start_temp, args.end_temp, config, iteration_per_cell, current_iteration, batch_size, total_iterations, in_auto_temp_schedule, const_temp, offset=True))

        try:
            # sometimes this doesn't catch the errors
            dask.distributed.wait(futures, 3600)
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
            results = sorted(results, key=lambda x: x[0], reverse=args.strategy == 'worst-wins' and not (current_iteration >= total_iterations))  # always sort by ascending cost when last batch
        cost, temp_lineage, temp_synthimages, temp_distmaps, temp_cellmaps = results[0]
        # print(f'Chose instance with cost: {cost}')

        # replace slices
        lineage.frames[window_start:window_end] = temp_lineage.frames
        realimages[window_start:window_end] = temp_realimages
        synthimages[window_start:window_end] = temp_synthimages
        cellmaps[window_start:window_end] = temp_cellmaps
        current_iteration += batch_size

        # update total_iterations
        total_iterations = iteration_per_cell * lineage.count_cells_in(window_start, window_end) // window

    # print(f'\nfinal cost: {cost}\n')
    return lineage, synthimages, distmaps, cellmaps


def optimize_core(lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, start_temp, end_temp, config,
                  iteration_per_cell, current_iteration, batch_size, total_iterations, in_auto_temp_schedule, const_temp, offset=False):

    if in_auto_temp_schedule:
        pbad_total = 0
        circular_buffer_capacity = config["pbad_max_size"]
        circular_buffer = np.empty(circular_buffer_capacity, float)
        circular_buffer_cursor = 0
        bad_count = 0

    window = window_end - window_start

    perturbation_prob = config["prob.perturbation"]
    combine_prob = config["prob.combine"]
    split_prob = config["prob.split"]
    background_offset_prob = config["prob.background_offset"]
    opacity_diffraction_offset_prob = config["prob.opacity_diffraction_offset"]
    camera_shift_prob = config["prob.camera_shift"]

    run_iterations = min(current_iteration + batch_size, total_iterations)
    while current_iteration < run_iterations:
        frame_index = lineage.choose_random_frame_index(window_start - window_start * offset, window_end - window_start * offset)  # offset true for multithreading
        if in_auto_temp_schedule:
            temperature = const_temp
        else:
            frame_start_temp = gerp(end_temp, start_temp, (frame_index - window_start * (not offset) + 1) / window)
            frame_end_temp = gerp(end_temp, start_temp, (frame_index - window_start * (not offset)) / window)
            temperature = gerp(frame_start_temp, frame_end_temp, current_iteration / (total_iterations))
        frame = lineage.frames[frame_index]
        node = np.random.choice(frame.nodes)
        if node.cell.dormant:
            continue
        change_option = np.random.choice(["split", "perturbation", "combine", "background_offset", "opacity_diffraction_offset", "camera_shift"],
                                         p=[split_prob, perturbation_prob, combine_prob, background_offset_prob, opacity_diffraction_offset_prob, camera_shift_prob])
        change = None
        if change_option == "split" and np.random.random_sample() < optimization.split_proba(node.cell.length) and not (window_start <= 0 and frame_index <= 0):
            change = Split(node.parent, config, realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], lineage.frames[frame_index], distmaps[frame_index])

        elif change_option == "perturbation":
            change = Perturbation(node, config, realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], lineage.frames[frame_index], distmaps[frame_index])

        elif change_option == "combine" and not (window_start <= 0 and frame_index <= 0):
            change = Combination(node.parent, config, realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], lineage.frames[frame_index], distmaps[frame_index])

        elif change_option == "background_offset" and not (window_start <= 0 and frame_index <= 0) and config["simulation"]["image.type"] == "graySynthetic":
            change = BackGroundLuminosityOffset(lineage.frames[frame_index], realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], config)

        elif change_option == "opacity_diffraction_offset" and not (window_start <= 0 and frame_index <= 0) and config["simulation"]["image.type"] == "graySynthetic":
            change = OpacityDiffractionOffset(lineage.frames[frame_index], realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], config)

        elif change_option == "camera_shift" and not (window_start <= 0 and frame_index <= 0):
            change = CameraShift(lineage.frames[frame_index], realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], config)

        if change and change.is_valid:
            # apply if acceptable
            costdiff = change.costdiff

            if costdiff <= 0:
                acceptance = 1.0
            else:
                acceptance = np.exp(-costdiff / temperature)
                if in_auto_temp_schedule:
                    pbad_total += acceptance
                    if (bad_count >= circular_buffer_capacity):
                        pbad_total -= circular_buffer[circular_buffer_cursor]
                    else:
                        bad_count += 1
                    circular_buffer[circular_buffer_cursor] = acceptance
                    circular_buffer_cursor = (circular_buffer_cursor + 1) % circular_buffer_capacity

            if acceptance > np.random.random_sample():
                change.apply()
                # if type(change) == Split:
                #     total_iterations += iteration_per_cell

                # if type(change) == Combination:
                #     total_iterations -= iteration_per_cell

        current_iteration += 1

    if in_auto_temp_schedule:
        print("pbad is ", pbad_total / bad_count)
        return pbad_total / bad_count

    realimage = realimages[0]
    shape = realimages[0].shape
    synthimage, cellmap = optimization.generate_synthetic_image(lineage.frames[-1].nodes, shape, lineage.frames[-1].simulation_config)
    cost = optimization.objective(realimage, synthimage, cellmap, config["overlap.cost"], config["cell.importance"])
    return cost, lineage, synthimages, distmaps, cellmaps


def optimize_old(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                 iteration_per_cell, in_auto_temp_schedule=False, const_temp=None):
    """Unchanged old optimize function to fall back to"""

    global totalCostDiff

    if in_auto_temp_schedule:
        lineage = deepcopy(lineage)
        synthimages = deepcopy(synthimages)
        distmaps = deepcopy(distmaps)
        cellmaps = deepcopy(cellmaps)

    debugfile = None
    if args.debug:
        debugfile = open(args.debug / 'debug.csv', 'a')

    pbad_total = 0
    circular_buffer_capacity = config["pbad_max_size"]
    circular_buffer = np.empty(circular_buffer_capacity, float)
    circular_buffer_cursor = 0

    perturbation_prob = config["prob.perturbation"]
    combine_prob = config["prob.combine"]
    split_prob = config["prob.split"]
    background_offset_prob = config["prob.background_offset"]
    opacity_diffraction_offset_prob = config["prob.opacity_diffraction_offset"]
    window = window_end - window_start

    # simulated annealing
    total_iterations = iteration_per_cell * lineage.count_cells_in(window_start, window_end) // window
    bad_count = 0
    current_iteration = 0
    while current_iteration < total_iterations:
        frame_index = lineage.choose_random_frame_index(window_start, window_end)
        if in_auto_temp_schedule:
            temperature = const_temp
        else:
            frame_start_temp = gerp(args.end_temp, args.start_temp, (frame_index - window_start + 1) / window)
            frame_end_temp = gerp(args.end_temp, args.start_temp, (frame_index - window_start) / window)
            temperature = gerp(frame_start_temp, frame_end_temp, current_iteration / total_iterations)
        frame = lineage.frames[frame_index]
        node = np.random.choice(frame.nodes)
        change_option = np.random.choice(["split", "perturbation", "combine", "background_offset", "opacity_diffraction_offset"], 
                                         p=[split_prob, perturbation_prob, combine_prob, background_offset_prob, opacity_diffraction_offset_prob])
        change = None
        if change_option == "split" and np.random.random_sample() < optimization.split_proba(node.cell.length) and frame_index > 0:
            change = Split(node.parent, config, realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], lineage.frames[frame_index], distmaps[frame_index])

        elif change_option == "perturbation":
            change = Perturbation(node, config, realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], lineage.frames[frame_index], distmaps[frame_index])

        elif change_option == "combine" and frame_index > 0:
            change = Combination(node.parent, config, realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], lineage.frames[frame_index], distmaps[frame_index])

        elif change_option == "background_offset" and frame_index > 0 and config["simulation"]["image.type"] == "graySynthetic":
            change = BackGroundLuminosityOffset(lineage.frames[frame_index], realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], config)

        elif change_option == "opacity_diffraction_offset" and frame_index > 0 and config["simulation"]["image.type"] == "graySynthetic":
            change = OpacityDiffractionOffset(lineage.frames[frame_index], realimages[frame_index], synthimages[frame_index], cellmaps[frame_index], config)

        if change and change.is_valid:
            # apply if acceptable
            costdiff = change.costdiff

            if costdiff <= 0:
                acceptance = 1.0
            else:
                acceptance = np.exp(-costdiff / temperature)
                pbad_total += acceptance
                if (bad_count >= circular_buffer_capacity):
                    pbad_total -= circular_buffer[circular_buffer_cursor]
                else:
                    bad_count += 1
                circular_buffer[circular_buffer_cursor] = acceptance
                circular_buffer_cursor = (circular_buffer_cursor + 1) % circular_buffer_capacity

            if acceptance > np.random.random_sample():
                totalCostDiff += costdiff
                change.apply()
                if type(change) == Split:
                    total_iterations += iteration_per_cell // window

                # if type(change) == Combination:
                #     total_iterations -= iteration_per_cell

        if debugfile and not in_auto_temp_schedule:
            print("{},{},{},{},{},{},{},{}".format(window_start, window_end, pbad_total, bad_count, temperature, totalCostDiff, current_iteration, total_iterations), file=debugfile)
        current_iteration += 1
        # print(current_iteration, total_iterations)

    if in_auto_temp_schedule:
        print("pbad is ", pbad_total / bad_count)
        return pbad_total / bad_count

    if debugfile:
        debugfile.close()

        # output module
