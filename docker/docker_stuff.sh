


IMG_ID=""
CONTAINER_NAME=""

function create_container() {
    USER_NAME=`whoami`
## Code snippet With display enabled
docker run --name $CONTAINER_NAME  \
        --gpus all  \
        -itd        \
        -v /home/${USER_NAME}:/home/${USER_NAME}       \
		-e DISPLAY=$DISPLAY                            \
        -e NVIDIA_DRIVER_CAPABILITIES=all              \
        --runtime=nvidia  \
        --shm-size 2G     \
        $IMG_ID           \
        bash
}


function enter_container() {
    docker exec -it "$CONTAINER_NAME" /bin/bash
}


function delete_container() {
    container_name="$1"
    docker rm "$container_name"
    echo "Deleted Docker container: $container_name"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--enter)
            enter_container
            shift 2
            ;;
        -c|--create)
            create_container
            shift
            ;;
        -d|--delete)
            container_name="$2"
            delete_container "$container_name"
            shift 2
            ;;
        *)
            echo "Invalid argument: $1"
            shift
            ;;
    esac
done

