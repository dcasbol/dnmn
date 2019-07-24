# Decoupled Neural Module Networks

A decoupled version of the Neural Module Networks architecture. Modules are trained independently and assembled afterwards. This code has been inspired and takes some code from Jacob Andreas' [original repository](https://github.com/jacobandreas/nmn2) too.

These experiments work on the VQA v1.0 dataset, which should be placed under the `data/vqa` directory. Image folders must be divided into `raw` and `conv` e.g. `Images/train2014/raw` containing the raw image files and the `Images/train2014/conv` is where the 14x14x512 features from the VGG16 will be stored. SPS2 files are provided so that it isn't necessary to install and run the Stanford parser.

Working with Python 3.5 or higher.

![ULPGC](https://www.siani.es/files/multimedia/imagenes/web/logo_ulpgc.png) ![SIANI](https://www.siani.es/files/multimedia/imagenes/web/logo_header.png)
