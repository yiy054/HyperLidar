# HyperLiDAR: Adaptive Post-Deployment LiDAR Segmentation via Hyperdimensional Computing [![arXiv](https://img.shields.io/badge/arXiv-2207.12691-b31b1b?logo=arXiv&logoColor=green)]()

Code for our paper:
> **HyperLiDAR: Adaptive Post-Deployment LiDAR Segmentation via Hyperdimensional Computing**
> <br>Ivannia Gomez Moreno, Yi Yao, Ye Tian, Xiaofan Yu, Flavio Ponzina, Michael Sullivan, Jingyi Zhang, Mingyu Yang, Hun Seok Kim and Tajana Rosing<br>
> 

<!-- ## Abstract： -->

## HyperLiDAR System Overview (Three Stages)

The full deployment pipeline of HyperLiDAR consists of three stages[cite: 135]:

1.  **Stage 1: Cloud Pre-training of Feature Extractor** (Off-device): A lightweight, generalized Feature Extractor (FE) is pretrained using large-scale data in the cloud and then frozen[cite: 136, 137].
2.  **Stage 2: HyperLiDAR On-Device Adaptation** (Edge): Only the HDC class hypervectors are adapted/retrained on new, streaming LiDAR scans[cite: 102, 133]. [cite_start]This stage utilizes the **Buffer Selection Strategy** for efficiency[cite: 109].
3.  **Stage 3: HyperLiDAR Inference** (Edge): New scans are processed through the frozen FE, encoded into a query hypervector, and classified using a simple similarity check against the trained class hypervectors[cite: 113, 128, 129].

## Dataset:
Download SemanticKITTI from [official web](http://www.semantic-kitti.org/dataset.html). 

## Prepare:
You can use the provided [Nautilus deployment YAML](./nautilus/hyperlidar.yaml) to launch the container environment.

- The base image used is:  
  ```
  ghcr.io/darthiv02/cenet_image:1.1
  ```
  > ⚠️ This image includes only the original **CENet** backbone and does **not** include HyperLiDAR support.

- To enable **TorchHD** functionality (for HyperLiDAR), you’ll need to manually install the `torch-hd` package [torch-hd](https://github.com/hyperdimensional-computing/torchhd?tab=readme-ov-file) :
  ```bash
  pip install torch-hd
  ```
- To work with **HyperLiDAR**, clone the specialized repository instead:
  ```bash
  git clone <this repo>
  ```

OR
- Feel free to used the `requirements.txt` to build the enviorment by yourself. 

## File Structure:
```
.
├── LICENSE
├── README.md           // this file
├── main.py             // main file (Hyperlidar method...)
├── methods             // implementation of HyperLidar/pretrainmodel..
├── requirements.txt
├── dataset             // dataset loader
├── kitti_pretrain       // pretrain model of semantickitti
└── config              // dataset and model config
```

## Usage：
### Train：
- **Stage 1: Cloud Pre-training of Feature Extractor:**
    This step pretrains the Feature Extractor before deployment. You can skip this step by using our provided pretrained model in `kitti_pretrain/`.
    ```
    python train.py 
      -d <dataset_path> 
      -ac config/arch/senet-512.yml 
      -l <save_dic> 
      -n senet-512 
      -p <pretrain_model_path> 
      -t <train_seqs>
    ```
    > **Note on Training Strategy:** For the pre-training, the model is first trained with 64x512 inputs. The pretrained model is then loaded for the subsequent Online learning phase on real-world sequences that have not been seen before. Current implementation focuses on the `senet-512` size.

- **Stage 2: HyperLiDAR On-Device Adaptation:**

  This runs the core HyperLiDAR adaptation, updating the HDC Class Hypervectors.

  Example Command (Using 1% Buffer Rate):

  `python main.py -d <path>/semantickitti -m <path>/HyperLidar/kitti_pretrain/senet-512 -t 5,6 -b 0.1 > ./result.log 2>&1`

  Example Command (Full Retraining - Equivalent to HyperLiDAR-full):
  `python main.py -d <path>/semantickitti -m <path>/HyperLidar/kitti_pretrain/senet-512 -t 5,6 -b 1 > ./result.log 2>&1`

  ```
  python main.py 
    -d <dataset_path> 
    -m <pretrain_model_path>
    -t <train_seqs>
    -b <buffer_rate>
  ```

## Pretrained Models and Logs:
| **KITTI Result** | 
| ---------------- | 
| kitti_pretrain | 


## Acknowledgments：
Code framework derived from [CENET](https://github.com/huixiancheng/CENet.git).

## Citation：
~~~

~~~
