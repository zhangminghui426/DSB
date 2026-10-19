FROM ubuntu:22.04

LABEL maintainer="zhangminghui"
LABEL description="DSB: bevfusion + flashocc + fast_calib development environment"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ========== COPY packages first (fail fast if missing) ==========
COPY packet/cuda_12.8.2_570.211.01_linux.run /tmp/cuda.run
COPY packet/cudnn-local-repo-ubuntu2204-9.26.1_1.0-1_amd64.deb /tmp/cudnn.deb
COPY packet/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.tar /tmp/tensorrt.tar
COPY packet/fish_install.yaml /tmp/fish_install.yaml

# ========== System dependencies ==========
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    sudo \
    wget \
    curl \
    vim \
    ca-certificates \
    software-properties-common \
    libssl-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libboost-all-dev \
    pkg-config \
    ninja-build \
    libncurses5 \
    libncursesw5 \
    && rm -rf /var/lib/apt/lists/*

# ========== CUDA 12.8.2 (runfile install) ==========
RUN sh /tmp/cuda.run --silent --toolkit --override --no-opengl-libs \
    && rm /tmp/cuda.run

ENV PATH=/usr/local/cuda-12.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda-12.8

# ========== cuDNN 9.26.1 (deb repo install, official NVIDIA commands) ==========
RUN dpkg -i /tmp/cudnn.deb \
    && cp /var/cudnn-local-repo-ubuntu2204-9.26.1/cudnn-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cudnn \
    && rm /tmp/cudnn.deb \
    && rm -rf /var/lib/apt/lists/*

# ========== TensorRT 10.9.0.34 (tarball install) ==========
RUN tar -xf /tmp/tensorrt.tar -C /opt \
    && mv /opt/TensorRT-10.9.0.34 /opt/tensorrt \
    && rm /tmp/tensorrt.tar

ENV TRT_HOME=/opt/tensorrt
ENV PATH=${TRT_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${TRT_HOME}/lib:${LD_LIBRARY_PATH}

# ========== ROS2 Humble (fishros 一键安装, 国内源) ==========
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    python3 \
    python3-yaml \
    python3-distro \
    locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# fishros 一键安装 ROS2 humble
# fish_install.yaml 提供自动应答（按交互顺序 9 项），避免交互式输入
RUN wget -q http://fishros.com/install -O /tmp/fishros \
    && FISH_INSTALL_CONFIG=/tmp/fish_install.yaml bash /tmp/fishros \
    && rm -f /tmp/fishros /tmp/fish_install.yaml

# 补充安装 mcap/bag/cv_bridge/pcl 等工具包
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros2bag \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-rosbag2-transport \
    ros-humble-image-tools \
    ros-humble-cv-bridge \
    ros-humble-pcl-conversions \
    ros-humble-pcl-ros \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble
ENV ROS_ROOT=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/ros/humble/bin:${PATH}
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV CMAKE_PREFIX_PATH=/opt/ros/humble

# ========== Miniconda3 ==========
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda config --set auto_activate_base false

# ========== DSB conda environment (Python 3.10) ==========
RUN conda create -n DSB python=3.10 -y \
    && echo "source activate DSB" > ~/.bashrc

ENV CONDA_DEFAULT_ENV=DSB
ENV CONDA_PREFIX=${CONDA_DIR}/envs/DSB
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

# ========== TensorRT Python wheel (install into DSB env) ==========
RUN pip install --no-cache-dir /opt/tensorrt/python/tensorrt-10.9.0-cp310-none-linux_x86_64.whl

# ========== PyTorch 2.8 + CUDA 12.8 ==========
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ========== Basic Python libraries ==========
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pyyaml \
    tqdm \
    tensorboard \
    matplotlib \
    opencv-python \
    opencv-contrib-python \
    pillow \
    pandas \
    scikit-learn \
    scikit-image \
    einops \
    addict \
    yapf \
    flake8 \
    black \
    ipython \
    jupyter \
    streamlit \
    open3d \
    laspy \
    pytorch3d

# ========== spconv for SECOND / BEVFusion ==========
RUN pip install --no-cache-dir \
    spconv-cu120

# ========== Workspace ==========
WORKDIR /workspace
COPY . /workspace

# ========== Default command ==========
CMD ["/bin/bash"]
