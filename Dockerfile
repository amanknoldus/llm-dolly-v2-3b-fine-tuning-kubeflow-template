FROM python:3.8

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
  graphviz


COPY requirements_x86.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy over and install source code from this package.
COPY src ./src
COPY utils ./utils
COPY constants.py ./constants.py
COPY README.md ./README.md
COPY setup.py ./setup.py
COPY version.txt ./version.txt

RUN pip install --no-cache-dir .