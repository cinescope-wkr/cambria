# syntax = devthefuture/dockerfile-x

RUN apt-get update && \
        apt-get install --no-install-recommends -y \
            wget && \
        apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ARG CONDA_PATH=/opt/conda
RUN ARCH=$(uname -m) && \
      if [ "${ARCH}" = "x86_64" ]; then \
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
      elif [ "${ARCH}" = "aarch64" ]; then \
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
      elif [ "${ARCH}" = "ppc64le" ]; then \
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh"; \
      else \
            echo "Unsupported architecture: ${ARCH}"; exit 1; \
      fi && \
      mkdir -p ${CONDA_PATH} && \
      wget ${MINICONDA_URL} -O ${CONDA_PATH}/miniconda.sh && \
      bash ${CONDA_PATH}/miniconda.sh -b -u -p ${CONDA_PATH} && \
      rm -rf ${CONDA_PATH}/miniconda.sh

ENV PATH "${CONDA_PATH}/bin:${PATH}"
ARG PATH "${CONDA_PATH}/bin:${PATH}"

# Create a conda environment for the project
USER ${USERNAME}
RUN conda init ${USERSHELL} && \
      conda create -n ${PROJECT} -y && \
      echo "conda activate ${PROJECT}" >> ${USERSHELLPROFILE}
ARG CONDA_CHANNELS=""
RUN [ -z "${CONDA_CHANNELS}" ] || \
      for channel in ${CONDA_CHANNELS}; do conda config --add channels ${channel}; done
ARG CONDA_PACKAGES=""
RUN [ -z "${CONDA_PACKAGES}" ] || conda install -n ${PROJECT} ${CONDA_PACKAGES} -y
USER root

# chown the conda dir so that we can edit it
RUN chown -R ${USERNAME}:${USERNAME} ${CONDA_PATH}
