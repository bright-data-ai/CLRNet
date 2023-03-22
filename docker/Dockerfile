FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY . /clrnet/
RUN conda env update -f /clrnet/environment.yml
RUN pip install /clrnet/docker/nms_impl-1.0-cp38-cp38-linux_x86_64.whl

ENV CUDA_HOME='/usr/local/cuda'
WORKDIR /clrnet
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]




