FROM python/3.9-slim

# Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip

RUN git clone https://github.com/washaq00/PointCloud_2Dto3D.git

RUN cd PointCloud_2Dto3D && python3 -m pip install -r requirements.txt && python3 -m pip install --upgrade pip && pip3 install torch torchvision torchaudio && python3 -m pip install geomloss

# Set the working directory
WORKDIR /PointCloud_2Dto3D

# Set the entrypoint
ENTRYPOINT [ "python3" ]