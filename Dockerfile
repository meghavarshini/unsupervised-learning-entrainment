ARG VERSION=20.04

FROM ubuntu:$VERSION

RUN apt-get update -y && apt-get install -y gnupg wget python3 python3-distro sudo && \
        apt-get -y install python3-pip python3-venv \
        && pip3 install setuptools build watermark

COPY ./setup.py /home/ubuntu/

RUN useradd ubuntu && \
    chown -R ubuntu:ubuntu /home/ubuntu

USER ubuntu
