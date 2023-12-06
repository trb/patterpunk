FROM ubuntu:23.10

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y \
    git \
    python3 \
    python3-pip \
    python-is-python3 && \
    rm -rf /var/lib/apt/lists/* \
    rm -f /usr/lib/python3.11/EXTERNALLY-MANAGED \
    rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

RUN mkdir -p /app/patterpunk /app/test

VOLUME /app/patterpunk
VOLUME /app/test

ENTRYPOINT ["/bin/bash"]