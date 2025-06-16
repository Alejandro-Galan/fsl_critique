name_im="ssl-symbols_3"
cont_im="ssl-symbols"
gpu=1
gpu=$1

docker run  -it -u $(id -u):$(id -g) --shm-size=1G --name $name_im -v $(pwd)/../:/home/user --rm --gpus device=$gpu $cont_im /bin/bash
#docker run  -it -u $(id -u):$(id -g) --name $name_im --log-driver local -v $(pwd)/logs:/var/log -v $(pwd)/../:/home/user --rm  $name_im /bin/bash

