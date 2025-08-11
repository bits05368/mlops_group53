IMAGE=bits05368/iris-mlops-api:latest
CONTAINER_NAME=iris-mlops-api

echo "Pulling latest image..."
docker pull $IMAGE

echo "Stopping existing container..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

echo "Starting container..."
docker run -d -p 8000:8000 --name $CONTAINER_NAME $IMAGE

echo "Deployment complete."