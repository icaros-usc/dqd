
# Differentiable Quality Diversity

This repository is the official implementation of Differentiable Quality Diversity.

The project contains a modified version of [pyribs](https://pyribs.org) a quality diversity optimization library. All MEGA variants are implemented in pyribs. The `GradientEmitter` implements both the OG-MAP-Elites and the OMG-MEGA algorithms. The `GradientImprovementEmitter` implements the CMA-MEGA algorithm.

See `ribs/emitters/_gradient_emitter.py` and `ribs/emitters/_gradient_improvement_emitter.py`.

## Requirements

The project builds in [Anaconda](www.anaconda.com).

Here are the instructions to create the conda environment:

```bash
conda env create -f experiments/environment.yml

```

Next install the local copy of pyribs after activating conda:

```bash
conda activate dqdexps
pip3 install -e .[all]
```

## Pretrained Models

You can download the StyleGAN pretrained models from the StyleGAN [repo](https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq-1024x1024.for_g_all.pt). Place the `.pt` file in the folder `experiments/lsi_clip`.

CLIP automatically installs with the conda environment.


## Running Experiments

For each experiment you pick an identifier for the algorithm you want to run.

| Quality Diversity Algorithm | Identifier      |
| --------------------------- | --------------: |
| MAP-Elites                  | map_elites      |
| MAP-Elites (line)           | map_elites_line |
| CMA-ME                      | cma_me_imp      |
| OG-MAP-Elites               | og_map_elites   |
| OMG-MEGA                    | omg_mega        |
| CMA-MEGA                    | cma_mega        |
| CMA-MEGA (Adam)             | cma_mega_adam   |

### Linear Projection (sphere)

To run an experiment with MAP-Elites:

```bash
conda activate dqdexps
cd experiments/lin_proj

python3 lin_proj.py map_elites --objective sphere
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 lin_proj.py --help

```

### Linear Projection (Rastrigin)


To run an experiment with MAP-Elites:

```bash
conda activate dqdexps
cd experiments/lin_proj

python3 lin_proj.py map_elites --objective Rastrigin
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 lin_proj.py --help

```

### Arm Repertoire

To run an experiment with MAP-Elites:

```bash
conda activate dqdexps
cd experiments/arm

python3 arm.py map_elites
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 arm.py --help

```

### Latent Space Illumination (LSI)

To run an experiment with MAP-Elites:

```bash
conda activate dqdexps
cd experiments/lsi_clip

python3 lsi.py map_elites 
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 lsi.py --help

```


## Results

The following tables contain the reported results from the DQD paper.

### Linear Projection (sphere)

| Quality Diversity Algorithms  | QD-score    | Coverage   |
| ----------------------------  | ----------: | ---------: |
| MAP-Elites                    |  1.04       |  1.17%     |
| MAP-Elites (line)             | 12.21       | 14.32%     |
| CMA-ME                        |  1.08       |  1.21%     |
| OG-MAP-Elites                 |  1.52       |  1.67%     |
| OMG-MEGA                      | 71.58       | 92.09%     |
| CMA-MEGA                      | 75.29       |100.00%     |
| CMA-MEGA (Adam)               | 75.3        |100.00%     |

### Linear Projection (Rastrigin)

| Quality Diversity Algorithms  | QD-score    | Coverage   |
| ----------------------------  | ----------: | ---------: |
| MAP-Elites                    |  1.18       |  1.72%     |
| MAP-Elites (line)             |  8.12       | 11.79%     |
| CMA-ME                        |  1.21       |  1.76%     |
| OG-MAP-Elites                 |  0.83       |  1.26%     |
| OMG-MEGA                      | 55.90       | 77.00%     |
| CMA-MEGA                      | 62.54       |100.00%     |
| CMA-MEGA (Adam)               | 62.58       |100.00%     |

### Arm Repertoire 

| Quality Diversity Algorithms  | QD-score    | Coverage   |
| ----------------------------  | ----------: | ---------: |
| MAP-Elites                    |  1.97       |  8.06%     |
| MAP-Elites (line)             | 33.51       | 35.79%     |
| CMA-ME                        | 55.98       | 56.95%     |
| OG-MAP-Elites                 | 57.17       | 58.08%     |
| OMG-MEGA                      | 44.12       | 44.13%     |
| CMA-MEGA                      | 74.18       | 74.18%     |
| CMA-MEGA (Adam)               | 73.82       | 73.82%     |

### Latent Space Illumination (LSI)

| Quality Diversity Algorithms  | QD-score    | Coverage   |
| ----------------------------  | ----------: | ---------: |
| MAP-Elites                    | 13.88       | 23.15%     |
| MAP-Elites (line)             | 16.54       | 25.73%     |
| CMA-ME                        | 18.96       | 26.18%     |
| CMA-MEGA                      |  5.36       |  8.61%     |
| CMA-MEGA (Adam)               | 21.82       | 30.73%     |


See the paper and supplementary materials for full data and standard error bars.


## License

pyribs and this project are both released under the MIT License.

[pyribs MIT License](https://github.com/icaros-usc/pyribs/blob/master/LICENSE)
