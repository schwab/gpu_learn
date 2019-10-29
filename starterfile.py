#  docker run -it --rm -v $(realpath ~/notebooks):/tf/notebooks --runtime=nvidia -p 8888:8888 -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility  tensorflow/tensorflow:latest-gpu-py3-jupyter
import tensorflow as tf;  

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))