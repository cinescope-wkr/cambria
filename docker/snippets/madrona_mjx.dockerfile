# syntax = devthefuture/dockerfile-x

RUN apt-get update && \
        apt-get install --no-install-recommends -y \
            wget \
            gpg \
            software-properties-common \
            lsb-release \
            openjdk-17-jre-headless \
            libgles2-mesa-dev \
            libglfw3-dev \
            libxrandr-dev \
            libxinerama-dev \
            libxcursor-dev \
            libxi-dev && \
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
        apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
        apt-get update && \
        apt-get install --no-install-recommends -y kitware-archive-keyring && \
        rm /etc/apt/trusted.gpg.d/kitware.gpg && \
        apt-get install --no-install-recommends -y \
            git-core \
            build-essential \
            cmake && \
        apt-get clean all && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ARG MADRONA_MJX_DIR=/opt/madrona_mjx
RUN git clone https://github.com/shacklettbp/madrona_mjx.git ${MADRONA_MJX_DIR} && \
        cd ${MADRONA_MJX_DIR} && \
        git submodule update --init --recursive && \
        mkdir build && \
        cd build && \
        cmake -DMADRONA_REQUIRE_CUDA=ON -DMADRONA_REQUIRE_PYTHON=ON .. && \
        make -j && \
        cd .. && \
        pip install -e .
