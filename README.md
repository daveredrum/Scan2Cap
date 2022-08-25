# Scan2Cap: Context-aware Dense Captioning in RGB-D Scans

<p align="center"><img src="demo/Scan2Cap.gif" width="600px"/></p>

## Introduction

We introduce the task of dense captioning in 3D scans from commodity RGB-D sensors. As input, we assume a point cloud of a 3D scene; the expected output is the bounding boxes along with the descriptions for the underlying objects. To address the 3D object detection and description problems, we propose Scan2Cap, an end-to-end trained method, to detect objects in the input scene and describe them in natural language. We use an attention mechanism that generates descriptive tokens while referring to the related components in the local context. To reflect object relations (i.e. relative spatial relations) in the generated captions, we use a message passing graph module to facilitate learning object relation features. Our method can effectively localize and describe 3D objects in scenes from the ScanRefer dataset, outperforming 2D baseline methods by a significant margin (27.61% CiDEr<!-- -->@<!-- -->0.5IoU improvement).

Please also check out the project website [here](https://daveredrum.github.io/Scan2Cap/index.html).

For additional detail, please see the Scan2Cap paper:  
"[Scan2Cap: Context-aware Dense Captioning in RGB-D Scans](https://arxiv.org/abs/2012.02206)"  
by [Dave Zhenyu Chen](https://daveredrum.github.io/), [Ali Gholami](https://aligholami.github.io/), [Matthias Nießner](https://www.niessnerlab.org/members/matthias_niessner/profile.html) and [Angel X. Chang](https://angelxuanchang.github.io/)  
from [Technical University of Munich](https://www.tum.de/en/) and [Simon Fraser University](https://www.sfu.ca/).

## :star2: Benchmark Challenge :star2:
We provide the Scan2Cap Benchmark Challenge for benchmarking your model automatically on the hidden test set! Learn more at our [benchmark challenge website](https://kaldir.vc.in.tum.de/scanrefer_benchmark/benchmark_captioning).
After finishing training the model, please download [the benchmark data](http://kaldir.vc.in.tum.de/scanrefer_benchmark_data.zip) and put the unzipped `ScanRefer_filtered_test.json` under `data/`. Then, you can run the following script the generate predictions:
```shell
python benchmark/predict.py --folder <output_folder> --use_multiview --use_normal --use_topdown --use_relation --num_graph_steps 2 --num_locals 10
```
Note that the flags must match the ones set before training. The training information is stored in `outputs/<folder_name>/info.json`. The generated predictions are stored in `outputs/<folder_name>/pred.json`.
For submitting the predictions, please compress the `pred.json` as a .zip or .7z file and follow the [instructions](http://kaldir.vc.in.tum.de/scanrefer_benchmark/documentation) to upload your results.

### Local Benchmarking on Val Set

Before submitting the results on the test set to the official benchmark, you can also benchmark the performance on the val set. Run the following script to generate GTs for val set first:

```shell
python scripts/build_benchmark_gt.py --split val
```

> NOTE: don't forget to change the `DATA_ROOT` in `scripts/build_benchmark_gt.py`

Generate the predictions on val set:

```shell
python benchmark/predict.py --folder <output_folder> --use_multiview --use_normal --use_topdown --use_relation --num_graph_steps 2 --num_locals 10 --test_split val
```

Evaluate the predictions on the val set:

```shell
python benchmark/eval.py --split val --path <path to predictions> --verbose
```

## Data

### ScanRefer

If you would like to access to the ScanRefer dataset, please fill out [this form](https://forms.gle/aLtzXN12DsYDMSXX6). Once your request is accepted, you will receive an email with the download link.

> Note: In addition to language annotations in ScanRefer dataset, you also need to access the original ScanNet dataset. Please refer to the [ScanNet Instructions](data/scannet/README.md) for more details.

Download the dataset by simply executing the wget command:
```shell
wget <download_link>
```

### Scan2CAD

As learning the relative object orientations in the relational graph requires CAD model alignment annotations in Scan2CAD, please refer to the [Scan2CAD official release](https://github.com/skanti/Scan2CAD#download-scan2cad-dataset-annotation-data) (you need ~8MB on your disk). Once the data is downloaded, extract the zip file under `data/` and change the path to Scan2CAD annotations (`CONF.PATH.SCAN2CAD`) in `lib/config.py` . As Scan2CAD doesn't cover all instances in ScanRefer, please download the [mapping file](http://kaldir.vc.in.tum.de/aligned_cad2inst_id.json) and place it under `CONF.PATH.SCAN2CAD`. Parsing the raw Scan2CAD annotations by the following command:

```shell
python scripts/Scan2CAD_to_ScanNet.py
```

## Setup

Please execute the following command to install PyTorch 1.8:

```shell
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

Install the necessary packages listed out in `requirements.txt`:
```shell
pip install -r requirements.txt
```

__And don't forget to refer to [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric#pytorch-180181) to install the graph support.__

After all packages are properly installed, please run the following commands to compile the CUDA modules for the PointNet++ backbone:
```shell
cd lib/pointnet2
python setup.py install
```
__Before moving on to the next step, please don't forget to set the project root path to the `CONF.PATH.BASE` in `lib/config.py`.__

### Data preparation
1. Download the ScanRefer dataset and unzip it under `data/` - You might want to run `python scripts/organize_scanrefer.py` to organize the data a bit. 
2. Download the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset).
> After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`
4. Pre-process ScanNet data. A folder named `scannet_data/` will be generated under `data/scannet/` after running the following command. Roughly 3.8GB free space is needed for this step:
```shell
cd data/scannet/
python batch_load_scannet_data.py
```
> After this step, you can check if the processed scene data is valid by running:
> ```shell
> python visualize.py --scene_id scene0000_00
> ```
<!-- 5. (Optional) Download the preprocessed [multiview features (~36GB)](http://kaldir.vc.in.tum.de/enet_feats.hdf5) and put it under `data/scannet/scannet_data/`. -->
5. (Optional) Pre-process the multiview features from ENet. 

    a. Download [the ENet pretrained weights (1.4MB)](http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth) and put it under `data/`
    
    b. Download and decompress [the extracted ScanNet frames (~13GB)](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip).

    c. Change the data paths in `config.py` marked with __TODO__ accordingly.

    d. Extract the ENet features:
    ```shell
    python scripts/compute_multiview_features.py
    ```

    e. Project ENet features from ScanNet frames to point clouds; you need ~36GB to store the generated HDF5 database:
    ```shell
    python scripts/project_multiview_features.py --maxpool
    ```
    > You can check if the projections make sense by projecting the semantic labels from image to the target point cloud by:
    > ```shell
    > python scripts/project_multiview_labels.py --scene_id scene0000_00 --maxpool
    > ```

## Usage

### End-to-End training for 3D dense captioning

Run the following script to start the end-to-end training of Scan2Cap model using the multiview features and normals. For more training options, please run `scripts/train.py -h`:

```shell
python scripts/train.py --use_multiview --use_normal --use_topdown --use_relation --use_orientation --num_graph_steps 2 --num_locals 10 --batch_size 12 --epoch 50
```

The trained model as well as the intermediate results will be dumped into `outputs/<output_folder>`. For evaluating the model (@0.5IoU), please run the following script and change the `<output_folder>` accordingly, and note that arguments must match the ones for training:

```shell
python scripts/eval.py --folder <output_folder> --use_multiview --use_normal --use_topdown --use_relation --num_graph_steps 2 --num_locals 10 --eval_caption --min_iou 0.5
```

Evaluating the detection performance:

```shell
python scripts/eval.py --folder <output_folder> --use_multiview --use_normal --use_topdown --use_relation --num_graph_steps 2 --num_locals 10 --eval_detection
```

You can even evaluate the pretraiend object detection backbone:

```shell
python scripts/eval.py --folder <output_folder> --use_multiview --use_normal --use_topdown --use_relation --num_graph_steps 2 --num_locals 10 --eval_detection --eval_pretrained
```

If you want to visualize the results, please run this script to generate bounding boxes and descriptions for scene `<scene_id>` to `outputs/<output_folder>`:

```shell
python scripts/visualize.py --folder <output_folder> --scene_id <scene_id> --use_multiview --use_normal --use_topdown --use_relation --num_graph_steps 2 --num_locals 10
```

> Note that you need to run `python scripts/export_scannet_axis_aligned_mesh.py` first to generate axis-aligned ScanNet mesh files.

### 3D dense captioning with ground truth bounding boxes

For experimenting the captioning performance with ground truth bounding boxes, you need to extract the box features with a pre-trained extractor. The pretrained ones are already in `pretrained`, but if you want to train a new one from scratch, run the following script:

```shell
python scripts/train_maskvotenet.py --batch_size 8 --epoch 200 --lr 1e-3 --wd 0 --use_multiview --use_normal
```

The pretrained model will be stored under `outputs/<output_folder>`. Before we proceed, you need to move the `<output_folder>` to `pretrained/` and change the name of the folder to `XYZ_MULTIVIEW_NORMAL_MASKS_VOTENET`, which must reflect the features while training, e.g. `MULTIVIEW` -> `--use_multiview`.

After that, let's run the following script to extract the features for the ground truth bounding boxes. Note that the feature options must match the ones in the previous steps:

```shell
python scripts/extract_gt_features.py --batch_size 16 --epoch 100 --use_multiview --use_normal --train --val
```

The extracted features will be stored as a HDF5 database under `<your-project-root>/gt_<dataset-name>_features`. You need ~610MB space on your disk.

Now the box features are ready - we're good to go! Next step: run the following command to start training the dense captioning pipeline with the extraced ground truth box features:

```shell
python scripts/train_pretrained.py --mode gt --batch_size 32 --use_topdown --use_relation --use_orientation --num_graph_steps 2 --num_locals 10
```

For evaluating the model, run the following command:

```shell
python scripts/eval_pretrained.py --folder <ouptut_folder> --mode gt --use_topdown --use_relation --use_orientation --num_graph_steps 2 --num_locals 10 
```

### 3D dense captioning with pre-trained VoteNet bounding boxes

If you would like to play around with the pre-trained VoteNet bounding boxes, you can directly use the pre-trained VoteNet in `pretrained`. After picking the model you like, run the following command to extract the bounding boxes and associated box features:

```shell
python scripts/extract_votenet_features.py --batch_size 16 --epoch 100 --use_multiview --use_normal --train --val
```

Now the box features are ready. Next step: run the following command to start training the dense captioning pipeline with the extraced VoteNet boxes:

```shell
python scripts/train_pretrained.py --mode votenet --batch_size 32 --use_topdown --use_relation --use_orientation --num_graph_steps 2 --num_locals 10
```

For evaluating the model, run the following command:

```shell
python scripts/eval_pretrained.py --folder <ouptut_folder> --mode votenet --use_topdown --use_relation --use_orientation --num_graph_steps 2 --num_locals 10 
```

### Experiments on ReferIt3D

Yes, of course you can use the ReferIt3D dataset for training and evaluation. Simply download ReferIt3D dataset and unzip it under `data`, then run the following command to convert it to ScanRefer format:

```shell
python scripts/organize_referit3d.py
```

Then you can simply specify the dataset you would like to use by `--dataset ReferIt3D` in the aforementioned steps. Have fun!

### 2D Experiments

Please refer to [`Scan2Cad-2D`](Scan2Cap-2D) for more information.

## Citation
If you found our work helpful, please kindly cite our paper via:
```bibtex
@inproceedings{chen2021scan2cap,
  title={Scan2Cap: Context-aware Dense Captioning in RGB-D Scans},
  author={Chen, Zhenyu and Gholami, Ali and Nie{\ss}ner, Matthias and Chang, Angel X},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3193--3203},
  year={2021}
}
```

## License
Scan2Cap is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](LICENSE).

Copyright (c) 2021 Dave Zhenyu Chen, Ali Gholami, Matthias Nießner, Angel X. Chang
