FROM pytorch/pytorch:latest

RUN apt update
RUN apt-get update -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y
RUN apt install nano

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install torchinfo
RUN pip install pandas
RUN pip install fire sudo


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
