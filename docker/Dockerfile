FROM tensorflow/tensorflow

RUN apt-get update && apt-get install -y --no-install-recommends \
        language-pack-ja-base \
        language-pack-ja \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        chainer \
        tensorflow

WORKDIR /workspace
