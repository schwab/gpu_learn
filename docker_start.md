docker start tf_gpu
docker run -it --name tf_gpu -p 8889:8888 -v /mnt/ssd1tb/home/mcstar/pyimagesearch:/workspace --device /dev/nvidia0:/dev/nvidia0 --runtime=nvidia   tensorflow/tensorflow:latest-gpu-py3
docker exec -it tf_gpu bash