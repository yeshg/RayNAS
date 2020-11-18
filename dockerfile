from continuumio/miniconda3
RUN DEBIAN_FRONTEND=noninteractive apt update && apt install -y build-essential git graphviz
RUN conda install python pip pygraphviz pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
RUN pip install filelock scipy graphviz opencv-python ray[tune]
RUN git clone https://github.com/yeshg/RayNAS
# Once running in container, run:
# python main.py darts cnn --dataset cifar10 --layers 2 --cuda