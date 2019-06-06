# Decoupled Neural Module Networks

A decoupled version of the Neural Module Networks architecture. Modules are trained independently and assembled afterwards. This code has been inspired and takes some code too from [Jacob Andreas' original repository](https://github.com/jacobandreas/nmn2).

These experiments work on the VQA dataset, which should be placed under the data directory, following the structure below:

+ data
  + vqa
    + Annotations
       mscoco_train2014_annotations.json
       mscoco_val2014_annotations.json
    + Images
      + test2015
      + train2014
      + val2014
      (each one with conv & raw subdirectories)
      normalizers.npz
    + Questions
       OpenEnded_mscoco_test2015_questions.json
       OpenEnded_mscoco_test-dev2015_questions.json
       OpenEnded_mscoco_train2014_questions.json
       OpenEnded_mscoco_val2014_questions.json
       test2015.sps2
       test-dev2015.sps2
       train2014.sps2
       val2014.sps2

![ULPGC](https://www.siani.es/files/multimedia/imagenes/web/logo_ulpgc.png) ![SIANI](https://www.siani.es/files/multimedia/imagenes/web/logo_header.png)
