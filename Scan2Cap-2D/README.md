# scan2cap-2d

## Setup


### Step 1 - Download & Unzip the Required Files
* Download the `hdf5` databaes for one of the following viewpoint types:

1. Annotated Viewpoint Database (Oracle Boxes):
    * [db_annotated.h5](https://mega.nz/file/eIR2CToR#795fMBvYjL9bOu4KaF5egbEn8UctsOtvrw-Rt1a5QUI) (~3GB)
    
2. Annotated Viewpoint Database (MRCNN Detected Boxes):
    * [db_annotated_mrcnn.h5](https://mega.nz/file/KcAW3J6R#r6HYeDbsa3_oWyvc3t3W4Z-xKvJ4r66i8nhYOIqLNXw) (~0.5GB)

3. Estimated Viewpoint Database (3D-2D Backprojected)
    * [db_estimated.h5](https://mega.nz/file/fdYEVTwa#tvoAc2bBreaqU2i4rHeLvk7Ywzltaj6XzXTWP9wbJj0) (~2.5GB)

4. Bird's Eye Viewpoint Database (Top-Down)
    * [db_td.h5](https://mega.nz/file/SIB2QTSA#z0uEWi8vZpik6O-13vSSUJoSWVzUlRtfOWI4p2C11D4) (~11GB)

* Download the ScanRefer `train` and `validation` splits:
    * [ScanRefer Download](https://github.com/daveredrum/ScanRefer#dataset)

* Download the vocabulary, glove embeddings and word weights:
    * [glove.p](https://mega.nz/file/KNQ3HaZK#lw00WZLklEna5d5Ru-V0Wre3_zz6tcGM9Thhr5NDwAs)
    * [ScanRefer_vocabulary.json](https://mega.nz/file/3AIBlChB#l2Q0bcl03K0Ooe6W60DecRUnWWUHltZPxQQbHo40iKY)
    * [ScanRefer_vocabulary_weights.json](https://mega.nz/file/qJY3gYaR#2YgGmNJTdvIqUoO3d3wDzwdd_TZxPRJgDzyC46d9BEU)

* Pre-trained models: https://mega.nz/folder/PJARTYJC#JhlUw3zagS9ck402_5TyRw

and unzip the downloaded files to your desired location.
Each database contains `global features`, `object features`, `object bounding box`, `semantic label` and `object id` corresponding to each sample in the desired `ScanRefer` split. 

### Optional - Prepare Databases from Scratch
Alternatively, you can manually render color and instance masks and use the code provided in `preprocessing` to obtain these databases. Make sure to set `IMAGE` and `INSTANCE_MASK` paths in the `conf.py` file. Here is a quick guide on how to use `preprocessing` module:

```
python main.py --prep --exp_type $ET --dataset $DS --viewpoint $VP --box $BX
```
where variables can take the following permutations:

| $DS           | $VP  |  $BX  | Comments
|:-----| :-----| :-----|:-----|
| scanrefer | annotated| oracle | Extracts oracle bounding boxes, bounding box features and global features from annotated viewpoints.
| scanrefer | annotated| mrcnn | Extracts MaskRCNN detected bounding boxes, bounding box features and global features from annotated viewpoints. 
| scanrefer | estimated| votenet | Extracts votenet estimated bounding boxes, bounding box features and global features from estimated viewpoints. 
| scanrefer | topdown | oracle | Extracts bird's eye view bounding boxes, bounding box features and global features from bird's eye viewpoints. 
---

### Step 2 - Install Required Packages
Code was tested with the following settings:
```
python==3.6.10
cudatoolkit==10.0.130
pytorch==1.2.0
torchvision==0.4.0
```
To setup the enviroment, simply run the following:
```
conda create -n scan2cap2d --file reqs_conda.txt && 
conda activate scan2cap2d && pip install -r reqs_pip.txt
```

## Training and Evaluation
Set the following paths in `scan2cap-2d/lib/conf.py` based on your needs:

```
CONF.PATH.DATA_ROOT = '/home/user/data'
CONF.PATH.CODE_ROOT = '/home/user/code'
CONF.PATH.SCANNET_DIR = "/scannet/public/v2"
```
---
Command-line arguments to run the training and/or evaluation; permutations are the same as provided in the `preprocessing` step.

    ap.add_argument("--exp_type", default="nret", help="retrieval or nonretrieval")
    ap.add_argument("--dataset", default="scanrefer", help="scanrefer or referit")
    ap.add_argument("--viewpoint", default="annotated", help="annotated, estimated or bev")
    ap.add_argument("--box", default="oracle", help="oracle, mrcnn or votenet")

1. Training and Evaluation
```
python main.py --train --exp_type $ET --dataset $DS --viewpoint $VP --box $BX --model $MD --visual_feat $VF...
```

where `$MD='snt'` for the Show and Tell model, and `$MD='satnt'` for Top-down and Bottom-up Attention Model. Also, `$VF` can take any combination of `'GTC'`, where it corresponds to `GLOBAL`, `TARGET` and `CONTEXT` respectively. Note that `$MD='snt'` only allows for `'GT'`. By default, `$ET` is set to `'nret'` which stands for `Non-Retrieval`. To run a retrieval experiment use `$ET='ret'`. 

other options include:
```
 --batch_size 128 
 --num_workers 16 
 --val_step 1000    
 --lr 1e-3
 --wd 1e-5
 --seed 42
```

2. Evaluation Only
```
python main.py --eval --exp_type $ET --dataset $DS --viewpoint $VP --box $BX --folder $EN
```
where `$EN` is the experiment directory name.

---
## Reproducing the results
Here is a set of experiments reported in the Scan2Cap paper and the commands to reproduce them. Please refer to table 6 and 8 in our paper for experiment names:
https://arxiv.org/pdf/2012.02206.pdf.

For the M2 and M2-RL results, please refer to the official [Meshed-Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer).

| Experiment           | Command | CIDER | BLEU-4 | METEOR | ROUGLE-L
|:----------------| :-----| :------| :------| :------| :------|
| {G, A, -, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box oracle  --visual_feat 'G' --model 'snt' --folder exp2 --ckpt_path pretrained/ant_snt_g/model.pth`` | 51.65 | 13.53 | 20.36 | 46.92 | 
| {T, A, O, Retr} | ``python main.py --eval --exp_type ret --dataset scanrefer --viewpoint annotated --box oracle --visual_feat 'T' --folder exp3`` | 30.64 | 9.74 | 18.93 | 41.26 | 
| {T+C, A, O, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box oracle --visual_feat 'TC' --model 'satnt' --folder exp4 --ckpt_path pretrained/ant_td_tc/model.pth`` | 48.50 | 15 | 20.52 | 49.31 | 
| {G+T, A, O, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box oracle --visual_feat 'GT' --model 'snt' --folder exp5 --ckpt_path pretrained/ant_snt_gt/model.pth`` | 60.95 | 14.79 | 21.24 | 47.91 | 
| {G+T+C, A, O, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box oracle --visual_feat 'GTC' --model 'satnt' --folder exp6 --ckpt_path pretrained/ant_td_gtc/model.pth`` | 20.34 | 7.47 | 16.54 | 40.25 | 
| {T+C, A, 2DM, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box mrcnn --visual_feat 'TC' --model 'satnt' --folder exp7 --ckpt_path pretrained/ant_td_tc/model.pth`` | 27.00 | 12.32 | 46.49 | 18.61 | 
| {G+T, A, 2DM, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box mrcnn --visual_feat 'GT' --model 'snt' --folder exp8 --ckpt_path pretrained/ant_snt_gt/model.pth`` | 32.88 | 12.32 | 19.38 | 45.04 | 
| {G+T+C, A, 2DM, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box mrcnn --visual_feat 'GTC' --model 'satnt' --folder exp9 --ckpt_path pretrained/ant_td_gtc/model.pth`` | 11.04 | 5.67 | 15.55 | 37.29 | 
| {T+C, E, 3DV, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint estimated --box votenet --visual_feat 'TC' --model 'satnt' --folder exp10 --ckpt_path pretrained/ant_td_tc/model.pth`` | 37.81 | 13.70 | 19.84 | 48.23 | 
| {G+T, E, 3DV, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint estimated --box votenet --visual_feat 'GT' --model 'snt' --folder exp11 --ckpt_path pretrained/ant_snt_gt/model.pth`` | 39.46 | 12.22 | 19.65 | 44.62 | 
| {G+T+C, E, 3DV, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint estimated --box votenet --visual_feat 'GTC' --model 'satnt' --folder exp12 --ckpt_path pretrained/ant_td_gtc/model.pth`` | 15.26 | 6.60 | 16.00 | 38.87 |
| {G, BEV, O, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint bev --box oracle --visual_feat 'G' --model 'snt' --folder exp13 --ckpt_path pretrained/bev_g/model.pth`` | 27.31 | 12.02 | 18.70 | 46.82 | 
| {G+T, BEV, O, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint bev --box oracle --visual_feat 'GT' --model 'snt' --folder exp14 --ckpt_path pretrained/bev_gt/model.pth`` | 30.41| 13.89 | 19.37 | 48.18| 
---