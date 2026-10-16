FROM nvcr.io/nvidia/tensorrt:25.09-py3

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
