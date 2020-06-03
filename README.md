# Design and Independent Training of Composable and Reusable Neural Modules

This is the code for the experiments shown in the paper _Design and Independent Training of Composable and Reusable Neural Modules_. All experiments can be replicated by running the scripts placed under the folder `experiments`.

The neural network used in the experiments is based on the Neural Module Networks architecture. Modules are trained independently and assembled afterwards. This code has been inspired and takes some code from Jacob Andreas' [original repository](https://github.com/jacobandreas/nmn2) too.

These experiments work on the VQA v1.0 dataset, which should be placed under the `data/vqa` directory. Image folders must be divided into `raw` and `conv` e.g. `Images/train2014/raw` containing the raw image files (they are usually placed under `Images/train2014`) and the `Images/train2014/conv` is where the 14x14x512 features from the VGG16 will be stored. SPS2 files are already provided in this repository, so that it isn't necessary to install and run the Stanford parser.

Working with Python 3.5 or higher.

[![ULPGC](https://www.siani.es/files/multimedia/imagenes/web/logo_ulpgc.png "ULPGC")](https://www.ulpgc.es/)          [![SIANI](https://www.siani.es/files/multimedia/imagenes/web/logo_header.png "SIANI")](https://www.siani.es/)
 
## Execution of experiments

All experiments listed here can be replicated just by running the corresponding script under the folder `experiments`. Please be sure to run all scripts from the repository's root folder. Before being able to run any experiment, follow these steps:

1. Ensure that you have put the VQA data as described before.
2. Ensure that you have virtualenv installed.
3. Run `00-setup.sh`. This will create a pair of virtual environments and preprocess input images.

### 01-validate_sparring.sh

Validation of the sparring module. Executes following steps:

1. Training of N=100 Find modules, using the sparring module for indirect supervision.
2. Filtering of trained Find modules, acording to uncertainty criteria, and selection of subset for correlation plot.
3. Test utility of each module by transferring to full NMN and training the rest.
4. Plot correlation found.


