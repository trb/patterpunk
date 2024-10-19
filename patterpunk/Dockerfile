FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN mkdir /app

RUN apt-get update \
  && apt-get full-upgrade -y \
  && apt-get install --no-install-recommends -y  \
    tzdata  \
    gnupg  \
    python3  \
    python3-pip \
    python-is-python3


COPY ./src/ /app

WORKDIR /app

RUN pip install --break-system-packages -r requirements.txt \
    && pip install --break-system-packages -r test.requirements.txt \
    && pip install --break-system-packages -r build.requirements.txt

VOLUME ["/app"]

ENTRYPOINT ["/bin/bash"]
