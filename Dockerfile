FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Basic updates. Do super early so we can cache for a long time
RUN apt update
RUN apt install -y curl

# Create a user to run under, and a cache directory for huggingface in it
#RUN useradd -m -d /server server
#USER server

#WORKDIR /server

RUN mkdir -p /huggingface

# Set up core python environment
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

COPY environment.yaml .
RUN bin/micromamba -r env -y create -f environment.yaml

# Install dependancies
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu116
ENV FLIT_ROOT_INSTALL=1

# We copy only the minimum for flit to run so avoid cache invalidation on code changes
COPY pyproject.toml .
COPY README.md .
COPY sdgrpcserver/__init__.py sdgrpcserver/
RUN bin/micromamba -r env -n sd-grpc-server run flit install --pth-file
RUN bin/micromamba -r env -n sd-grpc-server run pip cache purge

# Setup NVM & Node for Localtunnel
ENV NVM_DIR=/nvm
ENV NODE_VERSION=16.18.0

RUN mkdir -p $NVM_DIR

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default

ENV NODE_PATH $NVM_DIR/versions/node/v$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

RUN npm install -g localtunnel

# Now we can copy everything we need
COPY sdgrpcserver sdgrpcserver/
COPY server.py .

# Set up some config files
RUN mkdir -p weights
RUN mkdir -p config
COPY engines.yaml config/

# Set up some environment files
ENV HF_HOME=/huggingface
ENV HF_API_TOKEN=mustset
ENV SD_ENGINECFG=/config/engines.yaml
ENV SD_WEIGHT_ROOT=/weights

CMD [ "/bin/micromamba", "-r", "env", "-n", "sd-grpc-server", "run", "python", "./server.py" ]
