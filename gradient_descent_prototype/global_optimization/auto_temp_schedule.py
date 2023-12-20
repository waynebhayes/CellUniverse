from global_optimization import global_optimize


def auto_temp_schedule(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config):
    initial_temp = 1
    iteration_per_cell = config["auto_temp_scheduler.iteration_per_cell"]
    count = 0

    while(global_optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                   iteration_per_cell=iteration_per_cell, in_auto_temp_schedule=True, const_temp=initial_temp) < 0.3):
        count += 1
        initial_temp *= 10.0
    while(global_optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                   iteration_per_cell=iteration_per_cell, in_auto_temp_schedule=True, const_temp=initial_temp) > 0.3):
        count += 1
        initial_temp /= 10.0
    while(global_optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                   iteration_per_cell=iteration_per_cell, in_auto_temp_schedule=True, const_temp=initial_temp) < 0.3):
        count += 1
        initial_temp *= 1.1
    end_temp = initial_temp
    while(global_optimize(imagefiles, lineage, realimages, synthimages, cellmaps, distmaps, window_start, window_end, lineagefile, args, config,
                   iteration_per_cell=iteration_per_cell, in_auto_temp_schedule=True, const_temp=end_temp) > 1e-10):
        count += 1
        end_temp /= 10.0

    return initial_temp, end_temp
