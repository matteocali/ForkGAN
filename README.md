# ForkGAN: Readaptation by MEDIALab

## Description

This fork specializes in the use of **ForkGAN** for the readaptation of clear images of drones to night or rainy one.

To generate the night samples the model was trained on a combination of [Visdrone](https://github.com/VisDrone/VisDrone-Dataset) and [UAVDT](https://paperswithcode.com/dataset/uavdt), while for the rain samples the same model was drained on a combination of [BDD100K](https://bdd-data.berkeley.edu/), [ACDC](https://acdc.vision.ee.ethz.ch/) and [Cityscapes_rain](https://www.cityscapes-dataset.com/). In both cases the model was trained for 40 epochs

## Installation

In order to use the code create a new [conda](https://docs.conda.io/en/latest/) environment as follows:

```bash
conda create -n tf114 python=3.7
```

Then activate the environment and install the required packages:

```bash
conda activate tf114
conda install -f environment.yml
```

## Usage

### Datasets

If you want to train on the same datasets of use download them from the following links:

- [uavid]() (for testing),
- [multi_src]() (for night prediction),
- [bdd100k_acdc_synth]() (for rain prediction).

Put the downloaded (and extracted folder) inside the ```datasets``` folder in the root path of the code. The folder structure, for each dataset should be as follows:

```bash
├──datset_name
   ├── trainA (rainy or night images)
       ├── Image00001.jpg 
       └── ...
   ├── trainB (daytime images)
       ├── Image1.jpg
       └── ...
   ├── testA (testing rainy or night images)
       ├── Image13801.jpg (The test cover image that you want)
       └── ... 
   ├── testB (testing daytime images)
       ├── Image.jpg (The test message image that you want)
       └── ... 
```

### Training

To train the model on the night dataset run the following command from the root directory of the code:

```bash
bash scripts/night_train.sh
```

To train the model on the rain dataset run the following command from the root directory of the code:

```bash
bash scripts/rain_train.sh
```

### Testing

In order to test the model that we have trained download the pretrained models from the following links:

- [night model](),
- [rain model]().

Put the downloaded (and extracted folder) inside the ```check``` folder in the root path of the code.

To test the model on the night dataset run the following command from the root directory of the code:

```bash
bash scripts/night_test.sh
```

To test the model on the rain dataset run the following command from the root directory of the code:

```bash
bash scripts/rain_test.sh
```

## Acknowledgments

This repo is a fork of the one made by *Ziqiang Zheng*, *Yang Wu*, *Xinran Han*, and *Jianbo Shi*.

The original repo can be found [here](https://github.com/zhengziqiang/ForkGAN), while the linked paper can be found [here](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480154.pdf).

The modification of the original code was made by *[Matteo Caligiuri](https://github.com/matteocali/)*.
