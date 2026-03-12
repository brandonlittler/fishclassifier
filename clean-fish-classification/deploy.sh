
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACR_NAME="${ACR_NAME:-fishclassifieracr01}"
IMAGE="${ACR_NAME}.azurecr.io/fish-classifier:latest"

if [ ! -f "fish_classifier.pth" ]; then
  echo "⚠️  fish_classifier.pth not found. Creating placeholder so image builds; app will use randomly initialized weights."
  touch fish_classifier.pth
fi

echo "Building Docker image..."
docker build -t "$IMAGE" -f docker/Dockerfile .

echo "Logging in to ACR..."
az acr login --name "$ACR_NAME"

echo "Pushing image..."
docker push "$IMAGE"

echo "Deploying to Kubernetes..."
(cd kubernetes && ./deploy.sh)

pkill -f "port-forward.*8765:7860" 2>/dev/null || true
sleep 1
echo "Starting port-forward (localhost:8765 -> service:7860)..."
kubectl port-forward -n fish-classifier svc/fish-classifier-service 8765:7860 &
PF_PID=$!
sleep 2
if kill -0 $PF_PID 2>/dev/null; then
  echo "Done. Open http://127.0.0.1:8765 in your browser."
  echo "To stop port-forward later: kill $PF_PID"
else
  echo "Done. Port-forward may have failed; run ./port-forward.sh and open http://127.0.0.1:8765"
fi
