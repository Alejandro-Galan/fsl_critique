name_im="ssl-symbols_3"

docker build -t $name_im . --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
