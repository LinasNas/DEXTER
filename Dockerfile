FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive


# basic packages
RUN apt update && apt upgrade -y && apt install -y git vim wget sudo unzip ffmpeg xvfb

# python
RUN apt install -y python3 python3-pip python3-dev python3-tk

# packages for mujoco
RUN apt install -y \
    build-essential \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    gcc \
    && apt clean 
    #&& rm -rf /var/lib/apt/lists/*

#ADD BEFORE
RUN pip install Cython==3.0.0a10

#NEW: Install R
RUN apt-get update && \
    apt-get install -y r-base && \
    Rscript -e "install.packages('ocd', repos='http://cran.rstudio.com/')" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# MuJoCo + OpenAI gym
RUN pip install --upgrade "pip < 21.0"
RUN pip install gym==0.21.0
RUN mkdir -p /.mujoco && cd /.mujoco && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && tar -xf mujoco210-linux-x86_64.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH="/.mujoco/mujoco210"
ENV LD_LIBRARY_PATH /.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN pip install mujoco-py==2.1.2.14
RUN printf "import mujoco_py" | python3

# python packages
COPY ./requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# environment setup
RUN echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> /root/.bashrc
RUN echo 'alias python=python3' >> /root/.bashrc

#CMD /bin/bash
WORKDIR /home/duser/entryfolder
