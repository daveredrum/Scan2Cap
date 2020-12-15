# Scan2Cap: Context-aware Dense Captioning in RGB-D Scans

<p align="center"><img src="demo/Scan2Cap.gif" width="600px"/></p>

## Introduction

We introduce the task of dense captioning in 3D scans from commodity RGB-D sensors. As input, we assume a point cloud of a 3D scene; the expected output is the bounding boxes along with the descriptions for the underlying objects. To address the 3D object detection and description problems, we propose Scan2Cap, an end-to-end trained method, to detect objects in the input scene and describe them in natural language. We use an attention mechanism that generates descriptive tokens while referring to the related components in the local context. To reflect object relations (i.e. relative spatial relations) in the generated captions, we use a message passing graph module to facilitate learning object relation features. Our method can effectively localize and describe 3D objects in scenes from the ScanRefer dataset, outperforming 2D baseline methods by a significant margin (27.61% CiDEr<!-- -->@<!-- -->0.5IoU improvement).

Please also check out the project website [here](https://daveredrum.github.io/Scan2Cap/index.html).

For additional detail, please see the Scan2Cap paper:  
"[Scan2Cap: Context-aware Dense Captioning in RGB-D Scans](https://arxiv.org/abs/2012.02206)"  
by [Dave Zhenyu Chen](https://daveredrum.github.io/), [Ali Gholami](https://aligholami.github.io/), [Matthias Nießner](https://www.niessnerlab.org/members/matthias_niessner/profile.html) and [Angel X. Chang](https://angelxuanchang.github.io/)  
from [Technical University of Munich](https://www.tum.de/en/) and [Simon Fraser University](https://www.sfu.ca/).

## Coming soon!
