# diffusion-models

This repo provides a clean implementation of various types of diffusion models. Many of these are experimental research prototypes and represent work in progress.

## Structure

```
.
├── data                  # Utilities to load datasets
└── diffusion             # Implements various types of diffusion processes
  ├── gaussian.py         # Classical Gaussian diffusion
  ├── infomax.py          # Auxiliary-variable and information maximizing models (controllable diffusion models with small disentangled latents; experimental & in progress)
  └── learned.py        # Diffusion models where the noising process is learned (experimental & in progress)
├── misc                  # Miscellaneous utilities, like evaluation
└── models                # Implementations of denoising models
  ├── unet                # Various Unet type architectures
  └── modules             # Modules needed by the denoising models (attentinon, resnets, etc.)
├── trainer               # Modules that executes diffusion model training
└── README.md
```

## Environment

This code was tested in a conda environment created using:

```
conda create --name name python=3.7 pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch
```

To use this package, simply clone the git repo:

```
git https://github.com/kuleshov-group/diffusion-models.git;
cd diffusion-models
```

## Training the models

Training runs can be launched via the `run.py` script.

```
usage: run.py train [-h] [--model {gaussian,infomax,learned}]
                    [--dataset {fashion-mnist,mnist}]
                    [--checkpoint CHECKPOINT] [-e EPOCHS]
                    [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
                    [--optimizer {adam}] [--folder FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --model {gaussian,infomax,learned}
                        type of ddpm model to run
  --dataset {fashion-mnist,mnist}
                        training dataset
  --checkpoint CHECKPOINT
                        path to training checkpoint
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train
  --batch-size BATCH_SIZE
                        training batch size
  --learning-rate LEARNING_RATE
                        learning rate
  --optimizer {adam}    optimization algorithm
  --folder FOLDER       folder where logs will be stored
```

For example, this command trains an standard gaussian diffusion model for 50 epochs:

```
python run.py train --model gaussian --dataset fashion-mnist --folder gaussian-run --epochs 50
```

This command trains an auxiliary-variable diffusion model:

```
python run.py train --model infomax --folder gaussian-run --epochs 50
```

## Evaluation

The launcher script can also be used to trigger model evaluation.

```
usage: run.py eval [-h] [--model {gaussian,infomax,learned}]
                   [--dataset {fashion-mnist,mnist}] --checkpoint CHECKPOINT
                   [--deterministic] [--sample SAMPLE]
                   [--interpolate INTERPOLATE] [--latents LATENTS]
                   [--folder FOLDER] [--name NAME]

optional arguments:
  -h, --help            show this help message and exit
  --model {gaussian,infomax,learned}
                        type of ddpm model to run
  --dataset {fashion-mnist,mnist}
                        training dataset
  --checkpoint CHECKPOINT
                        path to training checkpoint
  --deterministic       run in deterministic mode
  --sample SAMPLE       how many samples to draw
  --interpolate INTERPOLATE
                        how many samples to interpolate
  --latents LATENTS     how many points to visualize in latent space
  --folder FOLDER       folder where output will be stored
  --name NAME           name of the files that will be saved
```

For example, this command loads the weights of an infomax model from an existing checkpoint, generates a figure with 100 samples from the model, and creates another figure with 128 test set points in a 2d latent space, colored according to their class label.

```
python run.py eval --model infomax --folder infomax-run --sample 100 --latents 128 --checkpoint infomax-run/model-49.pth
```

## Acknowledgements

* Jonathan Ho's orginal OpenAI codebase
* Phil Wang's (lucidrains) codebase
* The Annotated Diffusion Model
* The Latent Diffusion codebase
