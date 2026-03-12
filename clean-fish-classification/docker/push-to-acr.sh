set -e


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

REGISTRY_NAME="${REGISTRY_NAME:-fishclassifieracr01}"
IMAGE_TAG=${1:-"fish-classifier:latest"}

IMAGE_TAG=$(echo "${IMAGE_TAG}" | sed 's/.*\.azurecr\.io\///')

ACR_LOGIN_SERVER="${REGISTRY_NAME}.azurecr.io"
FULL_IMAGE_NAME="${ACR_LOGIN_SERVER}/${IMAGE_TAG}"



echo "Building Docker image..."
docker build -f docker/Dockerfile -t fish-classifier:latest .

echo "Tagging image for ACR..."
docker tag fish-classifier:latest ${FULL_IMAGE_NAME}

echo "Logging into Azure Container Registry..."
az acr login --name ${REGISTRY_NAME}

echo "Pushing image to ACR..."
docker push ${FULL_IMAGE_NAME}

echo "Success! Image pushed to: ${FULL_IMAGE_NAME}"
