ARG VERSION=20.04

FROM ubuntu:$VERSION

RUN apt-get update -y && apt-get install -y gnupg wget python3 python3-distro

COPY https://raw.githubusercontent.com/meghavarshini/unsupervised-learning-entrainment/master/setup.py  /home/ubuntu/

RUN useradd ubuntu && \
    chown -R ubuntu:ubuntu /home/ubuntu

USER ubuntu
