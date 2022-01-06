FROM arm64v8/python:3.9
WORKDIR mlops-image-generation-example
COPY . .
EXPOSE 8888
ENV JUPYTER_ENABLE_LAB=yes
RUN apt-get update && \
    make install
ENTRYPOINT ["/bin/bash"]