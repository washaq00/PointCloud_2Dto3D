FROM nvidia/cuda:12.3.2-base-ubuntu20.04

# Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip

RUN git clone https://github.com/washaq00/PointCloud_2Dto3D.git

RUN cd PointCloud_2Dto3D && python3 -m pip install -r requirements.txt

RUN wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz && wget https://drive.google.com/open?id=1cfoe521iTgcB_7-g_98GYAqO553W8Y0g && wget https://drive.google.com/open?id=10FR-2Lbn55POB1y47MJ12euvobi6mgtc
RUN unzip ShapeNetRendering.tgz -d ./data/shapenet && unzip ShapeNet_pointclouds.zip -d ./data/shapenet && unzip splits.zip -d ./data/shapenet

# Upgrade pip
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install geomloss

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Set the working directory
WORKDIR /PointCloud_2Dto3D

# Set the entrypoint
ENTRYPOINT [ "python3" ]