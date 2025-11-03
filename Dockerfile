# CENET

FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND='noninteractive'

## apt installs
RUN apt-get update && \
    apt install -y git wget && \
    apt-get install ffmpeg libsm6 libxext6 -y

## install conda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh && \
    bash Anaconda3-2023.03-1-Linux-x86_64.sh -b

ENV PATH=$PATH:/root/anaconda3/bin/

## install repo / conda environment (repo instructions)
RUN git clone --depth=1 https://github.com/huixiancheng/CENet /root/CENET
RUN conda create -n cenet python=3.8 -y && \
    conda init bash

WORKDIR /root/CENET

COPY . .

# make RUN commands use the new environment
SHELL ["conda", "run", "--no-capture-output", "-n", "cenet", "/bin/bash", "-c"]

RUN yes | apt-get install p7zip-full && \ 
yes | apt-get install vim && \
yes | apt-get install tmux 

RUN echo 'conda activate cenet' >> /root/.bashrc

RUN yes | pip install -r requirements_cenet.txt && \
yes | pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118