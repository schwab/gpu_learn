FROM tensorflow/tensorflow:latest-gpu-py3
ADD . /developer
LABEL maintainer="sahilmalik@winsmarts.com"
RUN apt update
RUN apt install graphviz -y
RUN apt install python3-opencv -y 
RUN pip install pydot -y

# install python app requirements
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
COPY . /opt/app
