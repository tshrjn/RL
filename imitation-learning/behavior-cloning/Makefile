# Installation Scripts
install-conda:
	wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
	sudo bash ./Anaconda3-4.4.0-Linux-x86_64.sh
install-python3.5.2:
	conda create --name gym python=3.5.2
get-mujoco:
	wget -O mjpro150.zip https://www.roboti.us/download/mjpro150_linux.zip
	sudo apt-get install unzip
	rm -rf ~/.mujoco
	mkdir ~/.mujoco
	unzip mjpro150.zip -d ~/.mujoco/mjpro150
	rm -rf mjpro150.zip
	echo "now put your `mjkey.txt` file into ~/.mujoco/mjkey.txt"
	sudo vim ~/.mujoco/mjkey.txt
install-mujoco-dependencies:
	sudo apt-get update -q
	DEBIAN_FRONTEND=noninteractive sudo apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev
	sudo apt-get clean
	sudo rm -rf /var/lib/apt/lists/*
# install patch
	sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
	sudo chmod +x /usr/local/bin/patchelf

install-mujoco:
# installing with pip simply doesn't work.
#	sudo pip3 install -U 'mujoco-py==1.50.1.0'
	rm -rf mujoco-py
	git clone https://github.com/openai/mujoco-py.git
	bash -c "cd mujoco-py && source activate gym && python setup.py install"
