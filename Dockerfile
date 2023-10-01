FROM nvidia/cuda:10.2-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
# https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/6.0/GA_6.0.1.8/local_repos/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt6.0.1.8-ga-20191108_1-1_amd64.deb
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
RUN apt-get update  -y --fix-missing
RUN apt-get install -y --no-install-recommends
RUN apt-get install wget -y
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get install -y software-properties-common \
                       curl unrar unzip
# RUN apt-get install libnvinfer6=6.0.1-1+cuda10.2 -y
# RUN apt-get install libnvinfer-dev=6.0.1-1+cuda10.2 -y
# RUN apt-get install libnvinfer-plugin6=6.0.1-1+cuda10.2 -y
# RUN apt-get install -y libnvinfer8
# RUN apt-get install -y libnvinfer-dev
# RUN apt-get install -y libnvinfer-plugin8
RUN apt-get install -y git
RUN apt-get clean -y

# TensorRT
COPY /external/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt6.0.1.8-ga-20191108_1-1_amd64.deb /tmp/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt6.0.1.8-ga-20191108_1-1_amd64.deb
RUN dpkg -i /tmp/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt6.0.1.8-ga-20191108_1-1_amd64.deb
# RUN apt-key add /var/nv-tensorrt-cuda10.2-trt6.0.1.8-ga-20191108/7fa2af80.pub
RUN apt-get update
# RUN apt-get install tensorrt \
#                     python3-libnvinfer-dev -y
RUN rm -rf /tmp/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt6.0.1.8-ga-20191108_1-1_amd64.deb

# Python
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
    bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -p /miniconda -b  && \
    rm -rf Miniconda3-py37_4.12.0-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -c anaconda -y \
      python=3.7.2 pip

# JupyterLab
RUN conda install -c conda-forge jupyterlab

# Main frameworks
RUN pip install torch

# Install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
