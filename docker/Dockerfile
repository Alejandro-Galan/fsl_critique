#FROM pytorch/pytorch:latest
FROM nvcr.io/nvidia/pytorch:24.01-py3


RUN apt update
RUN apt-get update -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y
RUN apt install nano

RUN pip install --upgrade pip


COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r requirements.txt

RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install torchinfo
RUN pip install pandas
RUN pip install fire sudo
RUN apt install graphviz -y

RUN pip install opencv-python==4.8.0.74
RUN pip install opencv-contrib-python==4.8.0.74
#RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
RUN pip install k-means-constrained --no-deps


## User specifics

# Bilbo agalan
ARG USER_ID=1003
ARG GROUP_ID=1003


RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers #.d/90-user

ENV HOME=/home/user
RUN chmod 777 /home/user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user
