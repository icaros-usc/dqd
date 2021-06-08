
import os
import csv
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

from dask.distributed import Client, LocalCluster

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (GaussianEmitter, ImprovementEmitter, IsoLineEmitter, 
                           GradientEmitter, GradientImprovementEmitter)
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

def evaluate_grasp(joint_angles, link_lengths, calc_jacobians=True):
    
    n_dim = link_lengths.shape[0]
    objs = -np.var(joint_angles, axis=1)

    # Remap the objective from [-1, 0] to [0, 100]
    objs = (objs+1.0)*100.0
    
    # theta_1, theta_1 + theta_2, ...
    cum_theta = np.cumsum(joint_angles, axis=1)
    # l_1 * cos(theta_1), l_2 * cos(theta_1 + theta_2), ...
    x_pos = link_lengths[None] * np.cos(cum_theta)
    # l_1 * sin(theta_1), l_2 * sin(theta_1 + theta_2), ...
    y_pos = link_lengths[None] * np.sin(cum_theta)

    measures = np.concatenate(
        (
            np.sum(x_pos, axis=1, keepdims=True),
            np.sum(y_pos, axis=1, keepdims=True),
        ),
        axis=1,
    )
    
    obj_derivatives = None
    measure_derivatives = None
    if calc_jacobians:
    
        means = np.mean(joint_angles, axis=1)
        means = np.expand_dims(means, axis=1)
    
        base = n_dim * np.ones(n_dim)
        obj_derivatives = -2 * (joint_angles - means) / base
    
        sum_0 = np.zeros(len(joint_angles)) 
        sum_1 = np.zeros(len(joint_angles)) 

        measure_derivatives = np.zeros((len(joint_angles), 2, n_dim))
        for i in range(n_dim-1, -1, -1):
            sum_0 += -link_lengths[i] * np.sin(cum_theta[:, i])
            sum_1 += link_lengths[i] * np.cos(cum_theta[:, i])
            
            measure_derivatives[:, 0, i] = sum_0
            measure_derivatives[:, 1, i] = sum_1

    return objs, obj_derivatives, measures, measure_derivatives


def create_optimizer(algorithm, dim, link_lengths, seed):
    """Creates an optimizer based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        dim (int): Dimensionality of the sphere function.
        seed (int): Main seed or the various components.
    Returns:
        Optimizer: A ribs Optimizer for running the algorithm.
    """
    max_bound = np.sum(link_lengths)
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    initial_sol = np.zeros(dim)
    batch_size = 36
    num_emitters = 1

    # Create archive.
    if algorithm in [
            "map_elites", "map_elites_line", "cma_me_imp",
            "og_map_elites", "omg_mega", "cma_mega", "cma_mega_adam",
    ]:
        archive = GridArchive((100, 100), bounds, seed=seed)
    else:
        raise ValueError(f"Algorithm `{algorithm}` is not recognized")


    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else list(
        range(seed, seed + num_emitters))
    if algorithm in ["map_elites"]:
        emitters = [
            GaussianEmitter(archive,
                            initial_sol,
                            0.1,
                            batch_size=batch_size,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["map_elites_line"]:
        emitters = [
            IsoLineEmitter(archive,
                           initial_sol,
                           iso_sigma=0.1,
                           line_sigma=0.2,
                           batch_size=batch_size,
                           seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["og_map_elites"]:
        emitters = [
            GradientEmitter(archive,
                            initial_sol,
                            sigma0=0.1,
                            sigma_g=100.0,
                            measure_gradients=False,
                            normalize_gradients=False,
                            bounds=None,
                            batch_size=batch_size // 2,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["omg_mega"]:
        emitters = [
            GradientEmitter(archive,
                            initial_sol,
                            sigma0=0.0,
                            sigma_g=1.0,
                            measure_gradients=True,
                            normalize_gradients=True,
                            bounds=None,
                            batch_size=batch_size // 2,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_mega"]:
        emitters = [
            GradientImprovementEmitter(archive,
                            initial_sol,
                            sigma_g=0.05,
                            stepsize=1.0,
                            gradient_optimizer="gradient_ascent",
                            normalize_gradients=True,
                            selection_rule="mu",
                            bounds=None,
                            batch_size=batch_size - 1,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_mega_adam"]:
        emitters = [
            GradientImprovementEmitter(archive,
                            initial_sol,
                            sigma_g=0.05,
                            stepsize=0.002, 
                            gradient_optimizer="adam",
                            normalize_gradients=True,
                            selection_rule="mu",
                            bounds=None,
                            batch_size=batch_size - 1,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_imp"]:
        emitters = [
            ImprovementEmitter(archive,
                               initial_sol,
                               0.1,
                               batch_size=batch_size,
                               seed=s) for s in emitter_seeds
        ]

    return Optimizer(archive, emitters)

def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0.0, vmax=100.0)
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def run_experiment(algorithm,
                   trial_id,
                   dim=1000,
                   init_pop=100,
                   itrs=10000,
                   outdir="logs",
                   log_freq=1,
                   log_arch_freq=1000,
                   seed=None):
    
    # Create a directory for this specific trial.
    s_logdir = os.path.join(outdir, f"{algorithm}", f"trial_{trial_id}")
    logdir = Path(s_logdir)
    if not logdir.is_dir():
        logdir.mkdir()

    # Create a new summary file
    summary_filename = os.path.join(s_logdir, f"summary.csv")
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, 'w') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(['Iteration', 'QD-Score', 'Coverage', 'Maximum', 'Average'])

    link_lengths = np.ones(dim)
    
    is_init_pop = algorithm in ['og_map_elites', 'omg_mega', 'map_elites', 'map_elites_line']
    is_dqd = algorithm in ['og_map_elites', 'omg_mega', 'cma_mega', 'cma_mega_adam']

    optimizer = create_optimizer(algorithm, dim, link_lengths, seed)
    archive = optimizer.archive

    best = 0.0
    non_logging_time = 0.0
    with alive_bar(itrs) as progress:

        if is_init_pop:
            # Sample initial population
            sols = np.array([np.random.normal(size=dim) for _ in range(init_pop)])

            objs, _, measures, _ = evaluate_grasp(sols, link_lengths)
            best = max(best, max(objs))

            # Add each solution to the archive.
            for i in range(len(sols)):
                archive.add(sols[i], objs[i], measures[i])

        for itr in range(1, itrs + 1):
            itr_start = time.time()

            if is_dqd:
                sols = optimizer.ask(grad_estimate=True)
                objs, jacobian_obj, measures, jacobian_measure = evaluate_grasp(sols, link_lengths)
                best = max(best, max(objs))
                jacobian_obj = np.expand_dims(jacobian_obj, axis=1)
                jacobian = np.concatenate((jacobian_obj, jacobian_measure), axis=1)
                optimizer.tell(objs, measures, jacobian=jacobian)

            sols = optimizer.ask()
            objs, _, measures, _ = evaluate_grasp(sols, link_lengths, calc_jacobians=False)
            best = max(best, max(objs))
            optimizer.tell(objs, measures)
            non_logging_time += time.time() - itr_start
            progress()

            # Save the archive at the given frequency.
            # Always save on the final iteration.
            final_itr = itr == itrs
            if (itr > 0 and itr % log_arch_freq == 0) or final_itr:

                # Save a full archive for analysis.
                df = archive.as_pandas(include_solutions = final_itr)
                df.to_pickle(os.path.join(s_logdir, f"archive_{itr:05d}.pkl"))

                # Save a heatmap image to observe how the trial is doing.
                save_heatmap(archive, os.path.join(s_logdir, f"heatmap_{itr:05d}.png"))

            # Update the summary statistics for the archive
            if (itr > 0 and itr % log_freq == 0) or final_itr:
                with open(summary_filename, 'a') as summary_file:
                    writer = csv.writer(summary_file)

                    sum_obj = 0
                    num_filled = 0
                    num_bins = archive.bins
                    for sol, obj, beh, idx, meta in zip(*archive.data()):
                        num_filled += 1
                        sum_obj += obj
                    qd_score = sum_obj / num_bins
                    average = sum_obj / num_filled
                    coverage = 100.0 * num_filled / num_bins
                    data = [itr, qd_score, coverage, best, average]
                    writer.writerow(data)


def arm_main(algorithm,
             trials=20,
             dim=1000,
             init_pop=100,
             itrs=10000,
             outdir="logs",
             log_freq=1,
             log_arch_freq=1000,
             seed=None):
    """Experimental tool for the planar robotic arm experiments.

    Args:
        algorithm (str): Name of the algorithm.
        trials (int): Number of experimental trials to run.
        dim (int): Dimensionality of solutions.
        init_pop (int): Initial population size for MAP-Elites (ignored for CMA variants).
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations between computing QD metrics and updating logs.
        log_arch_freq (int): Number of iterations between saving an archive and generating heatmaps.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """

    # Create a shared logging directory for the experiments for this algorithm.
    s_logdir = os.path.join(outdir, f"{algorithm}")
    logdir = Path(s_logdir)
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()
    if not logdir.is_dir():
        logdir.mkdir()

    cluster = LocalCluster(
        processes=True,  # Each worker is a process.
        n_workers=trials,  # Create one worker per trial (assumes >=trials cores)
        threads_per_worker=1,  # Each worker process is single-threaded.
    )
    client = Client(cluster)

    exp_func = lambda cur_id: run_experiment(
            algorithm, cur_id,
            dim=dim,
            init_pop=init_pop,
            itrs=itrs,
            outdir=outdir,
            log_freq=log_freq,
            log_arch_freq=log_arch_freq,
            seed=seed,
        )

    # Run an experiment as a separate process to run all exps in parallel.
    trial_ids = list(range(trials))
    futures = client.map(exp_func, trial_ids)
    results = client.gather(futures)


if __name__ == '__main__':
    fire.Fire(arm_main)
