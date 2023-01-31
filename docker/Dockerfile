from ubuntu:jammy
run unset https_proxy
add ./sources.list /etc/apt/sources.list
add ./cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg
run apt-get update
ARG DEBIAN_FRONTEND=noninteractive
run apt-get install -y apt-utils
run apt-get install -y  ca-certificates
run apt-get install -y vim 

run echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list 
run apt-get update


run apt-get install -y wget curl vim gcc git openssl\
             htop atop screen locales tmux mc sudo

run apt-get install -y  cuda-toolkit-11.7
run apt-get install -y libcudnn8=8.5.0.96-1+cuda11.7
run apt-mark hold libcudnn8 

#run apt-get install -y nvidia-utils-520
run apt-get install -y pip  
run apt-get install -y expect  
run apt-get install -y zsh  
run apt-get install -y curl  

run locale-gen "en_US.UTF-8"
run dpkg-reconfigure locales
run update-locale LANG=en_US.UTF-8 LANGUAGE=en.UTF-8
run mkdir -p  /home/aprec/Projects/aprec
run useradd aprec
run chown -R aprec:aprec /home/aprec/
run usermod -aG sudo aprec
run usermod --password $(openssl passwd -1 lambdarank) aprec
run echo "aprec ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers 
user aprec

SHELL ["/bin/zsh", "-c"]
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.4/zsh-in-docker.sh)" -- \
    -t robbyrussell


ENV SHELL=/bin/zsh
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
add .vimrc /home/aprec/.vimrc 
add .gitconfig /home/aprec/.gitconfig 
#
##make /bin/sh symlink to zsh instead of dash:
user root
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash
user aprec
ENV ENV=/home/aprec/.profile
ENV PATH=$PATH:/home/aprec/.local/bin

workdir /home/aprec/Projects/
ENV PYTHONPATH=/home/aprec/Projects

add requirements.txt  /tmp/aprec_requirements.txt
run pip3 install -r /tmp/aprec_requirements.txt

cmd zsh
