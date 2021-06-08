
import os
import csv
import time
from pathlib import Path

import fire
import clip
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from alive_progress import alive_bar
from PIL import Image

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (GaussianEmitter, ImprovementEmitter, IsoLineEmitter, 
                           GradientEmitter, GradientImprovementEmitter)
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

from stylegan_models import g_all, g_synthesis, g_mapping


def tensor_to_pil_img(img):
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    return img

def compute_clip_loss(device, c_net, img, text):
    img = torch.nn.functional.upsample_bilinear(img, (224, 224))
    tokenized_text = clip.tokenize([text]).to(device)

    img_logits, _text_logits = c_net(img, tokenized_text)

    return 1/img_logits * 100

def compute_clip_losses(device, c_net, img, prompts):
    tokenized_text = clip.tokenize(prompts).to(device)

    img_logits, _text_logits = c_net(img, tokenized_text)

    return 1/img_logits * 100

def compute_prompts(device, latent_code, g_net, c_net, prompts, img_batch_size=37):
    

    imgs = []
    for i in range(0, len(latent_code), img_batch_size):
        
        latents = torch.nn.Parameter(latent_code[i:i+img_batch_size], requires_grad=False)
        dlatents = latents.repeat(1,18,1)
        
        img = g_net(dlatents)
        img = torch.nn.functional.upsample_bilinear(img, (224, 224))

        imgs.append(img)

    img = torch.cat(imgs)
    loss = compute_clip_losses(device, c_net, img, prompts)
    value = loss.cpu().detach().numpy()
    
    return value
    

def compute_value_jacobian(device, latent_code, g_net, c_net, text, calc_jacobian=True):

    latents = torch.nn.Parameter(latent_code, requires_grad=calc_jacobian)
    dlatents = latents.repeat(1,18,1)

    img = g_net(dlatents)

    loss = compute_clip_loss(device, c_net, img, text)

    value = loss.cpu().detach().numpy()
    value = np.squeeze(value, axis=1)

    jacobian = None
    if calc_jacobian:
        loss.backward()
        jacobian = latents.grad.cpu().detach().numpy()
        jacobian = np.squeeze(-jacobian, axis=0)

    return value, jacobian

def compute_values_jacobians(device, latent_code, g_net, c_net, texts, calc_jacobian=True):
    
    values = []
    jacobians = []

    for text in texts:
        value, jacobian = compute_value_jacobian(device, latent_code, g_net, c_net, 
                                                 text, calc_jacobian)
        values.append(value)
        jacobians.append(jacobian)

    jacobian = None
    if calc_jacobian:
        jacobian = np.array(jacobians)

    return np.array(values), jacobian

def transform_obj(objs):
    # Remap the objective from minimizing [0, 10] to maximizing [0, 100]
    return (10.0-objs)*10.0

def create_optimizer(algorithm, dim, seed):
    """Creates an optimizer based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        dim (int): Dimensionality of the sphere function.
        seed (int): Main seed or the various components.
    Returns:
        Optimizer: A ribs Optimizer for running the algorithm.
    """
    bounds = [(0.0, 6.0), (0.0, 6.0)]
    initial_sol = np.zeros(dim)
    batch_size = 36
    num_emitters = 1

    # Create archive.
    if algorithm in [
            "map_elites", "map_elites_line", "cma_me_imp",
            "og_map_elites", "omg_mega", "cma_mega", "cma_mega_adam",
    ]:
        archive = GridArchive((200, 200), bounds, seed=seed)
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
                            0.2,
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
                            sigma0=0.2,
                            sigma_g=0.2,
                            measure_gradients=False,
                            bounds=None,
                            batch_size=batch_size // 2,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["omg_mega"]:
        emitters = [
            GradientEmitter(archive,
                            initial_sol,
                            sigma0=0.0,
                            sigma_g=0.2,
                            measure_gradients=True,
                            bounds=None,
                            batch_size=batch_size // 2,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_mega"]:
        emitters = [
            GradientImprovementEmitter(archive,
                            initial_sol,
                            sigma_g=0.002,
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
                            sigma_g=0.002,
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
                               0.02,
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
    grid_archive_heatmap(archive, vmin=0, vmax=100)
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())

def run_experiment(algorithm,
                   trial_id,
                   clip_model,
                   generator,
				   device,
                   dim=512,
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

    is_init_pop = algorithm in ['og_map_elites', 'omg_mega', 'map_elites', 'map_elites_line']
    is_dqd = algorithm in ['og_map_elites', 'omg_mega', 'cma_mega', 'cma_mega_adam']

    optimizer = create_optimizer(algorithm, dim, seed)
    archive = optimizer.archive

    objective_prompt = 'Elon Musk with short hair.'
    measure_prompts = ['An man with blue eyes.', 'A person with red hair.']
    all_prompts = [objective_prompt] + measure_prompts

    best = -1000
    non_logging_time = 0.0
    with alive_bar(itrs) as progress:

        if is_init_pop:
            # Sample initial population
            sols = np.array([np.random.normal(size=dim) for _ in range(init_pop)])
            sols = np.expand_dims(sols, axis=1)
            latent_codes = torch.tensor(sols, dtype=torch.float32, device=device)

            values = compute_prompts(device, latent_codes, generator, clip_model, all_prompts)

            objs = values[:,0]
            measures = values[:,1:3]

            objs = transform_obj(np.array(objs, dtype=np.float32))
            measures = np.array(measures, dtype=np.float32)

            best_gen = max(objs) 
            best = max(best, best_gen)

            # Add each solution to the archive.
            for i in range(len(sols)):
                archive.add(sols[i], objs[i], measures[i])

        for itr in range(1, itrs + 1):
            itr_start = time.time()

            if is_dqd:
                sols = optimizer.ask(grad_estimate=True)
                nvec = np.linalg.norm(sols)

                latent_codes = torch.tensor(sols, dtype=torch.float32, device=device)

                objs, jacobian_obj = compute_value_jacobian(device, latent_codes, generator, 
                                                            clip_model, objective_prompt,
                                                            calc_jacobian=True)
                objs = transform_obj(objs)
                best = max(best, max(objs))

                measures, jacobian_measure = compute_values_jacobians(device, latent_codes,
                                             generator, clip_model, measure_prompts,
                                             calc_jacobian=True)

                jacobian_obj = np.expand_dims(jacobian_obj, axis=0)
                jacobian = np.concatenate((jacobian_obj, jacobian_measure), axis=0)
                jacobian = np.expand_dims(jacobian, axis=0)

                measures = np.transpose(measures)

                objs = objs.astype(np.float32)
                measures = measures.astype(np.float32)
                jacobian = jacobian.astype(np.float32)

                optimizer.tell(objs, measures, jacobian=jacobian)

            sols = optimizer.ask()
            sols = np.expand_dims(sols, axis=1)
            latent_codes = torch.tensor(sols, dtype=torch.float32, device=device)

            values = compute_prompts(device, latent_codes, generator, clip_model, all_prompts)

            objs = values[:,0]
            measures = values[:,1:3]

            objs = transform_obj(np.array(objs, dtype=np.float32))
            measures = np.array(measures, dtype=np.float32)

            best_gen = max(objs) 
            best = max(best, best_gen)

            optimizer.tell(objs, measures)
            non_logging_time += time.time() - itr_start
            progress()

            print('best', best, best_gen)

            # Save the archive at the given frequency.
            # Always save on the final iteration.
            final_itr = itr == itrs
            if (itr > 0 and itr % log_arch_freq == 0) or final_itr:

                # Save a full archive for analysis.
                df = archive.as_pandas(include_solutions = final_itr)
                df.to_pickle(os.path.join(s_logdir, f"archive_{itr:06d}.pkl"))

                # Save a heatmap image to observe how the trial is doing.
                save_heatmap(archive, os.path.join(s_logdir, f"heatmap_{itr:06d}.png"))

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


def lsi_main(algorithm, 
             trials=5,
             init_pop=100,
             itrs=10000,
             outdir='logs',
             log_freq=1,
             log_arch_freq=1000,
             seed=None):
    """Experimental tool for the StyleGAN+CLIP LSI experiments.

    Args:
        algorithm (str): Name of the algorithm.
        trials (int): Number of experimental trials to run.
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

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    
    g_synthesis.eval()
    g_synthesis.to(device)
    for p in g_synthesis.parameters():
        p.requires_grad_(False)

    # Latent space is size 512
    dim = 512

    for cur_id in range(trials):
        run_experiment(algorithm, cur_id, clip_model, g_synthesis,
                       device, dim=dim, init_pop=init_pop, itrs=itrs,
                       outdir=outdir, log_freq=log_freq, 
                       log_arch_freq=log_arch_freq, seed=seed)
     
if __name__ == '__main__':
    fire.Fire(lsi_main)
