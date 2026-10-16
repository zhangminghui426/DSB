FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="zhangminghui"
LABEL description="DSB: bevfusion + flashocc + fast_calib development environment"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ========== System dependencies ==========
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
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
    && rm -rf /var/lib/apt/lists/*

# ========== ROS2 Humble (Ubuntu 22.04) ==========
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
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
    && rm -rf /var/lib/apt/lists/* \
    && rosdep init || echo "rosdep already initialized" \
    && rosdep update --rosdistro humble

ENV ROS_DISTRO=humble
ENV ROS_ROOT=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/ros/humble/bin:${PATH}
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV CMAKE_PREFIX_PATH=/opt/ros/humble

# ========== TensorRT 10.9 (tarball install) ==========
COPY TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz /tmp/tensorrt.tar.gz
RUN tar -xzf /tmp/tensorrt.tar.gz -C /opt \
    && mv /opt/TensorRT-10.9.0.34 /opt/tensorrt \
    && rm /tmp/tensorrt.tar.gz \
    && pip install --no-cache-dir /opt/tensorrt/python/tensorrt-10.9.0-cp310-none-linux_x86_64.whl

ENV TRT_HOME=/opt/tensorrt
ENV PATH=${TRT_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${TRT_HOME}/lib:${LD_LIBRARY_PATH}

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
