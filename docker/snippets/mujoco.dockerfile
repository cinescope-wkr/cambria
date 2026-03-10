# syntax = devthefuture/dockerfile-x

RUN apt-get update && \
        apt-get install --no-install-recommends -y \
        git \
        python3-venv \
        libgl1-mesa-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxrandr-dev \
        libxi-dev \
        build-essential \
        cmake \
        ninja-build && \
        apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ARG MUJOCO_DIR=/opt/mujoco
RUN git clone https://github.com/google-deepmind/mujoco.git ${MUJOCO_DIR} && \
        mkdir ${MUJOCO_DIR}/build && \
        cd ${MUJOCO_DIR}/build && \
        cmake .. -G Ninja -DCMAKE_INSTALL_PREFIX=${MUJOCO_DIR}/install -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=TRUE && \
        ninja && \
        ninja install

RUN cd ${MUJOCO_DIR}/python && \
        python3 -m venv /tmp/env && \
        . /tmp/env/bin/activate && \
        bash make_sdist.sh && \
        deactivate && \
        MUJOCO_PATH=${MUJOCO_DIR}/install MUJOCO_PLUGIN_PATH=../../plugin pip install dist/mujoco*.tar.gz && \
        rm -rf /tmp/env
