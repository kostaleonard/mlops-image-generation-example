FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3
WORKDIR mlops-image-generation-example
COPY . .
EXPOSE 8888
ENV JUPYTER_ENABLE_LAB=yes
RUN apt-get update && \
    apt-get install -y vim && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt
ENTRYPOINT ["/bin/bash"]
