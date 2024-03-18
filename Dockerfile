FROM nvidia/cuda:12.3.2-base-ubuntu20.04

# Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip

RUN ls

RUN git clone https://github.com/washaq00/PointCloud_2Dto3D.git

RUN cd PointCloud_2Dto3D && python3 -m pip install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip && python3 -m pip install geomloss

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Set the working directory
WORKDIR /PointCloud_2Dto3D

# Set the entrypoint
ENTRYPOINT [ "python3" ]