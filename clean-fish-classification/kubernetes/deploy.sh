set -e

SUBSCRIPTION_ID="${SUBSCRIPTION_ID:?Set SUBSCRIPTION_ID}"
ACR_NAME="${ACR_NAME:?Set ACR_NAME}"
RESOURCE_GROUP="${RESOURCE_GROUP:?Set RESOURCE_GROUP}"
NAMESPACE="${NAMESPACE:-fish-classifier}"
SECRET_NAME="${SECRET_NAME:-acr-secret}"

az account set --subscription "${SUBSCRIPTION_ID}"

az acr update -n "${ACR_NAME}" --admin-enabled true

ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query "passwords[0].value" -o tsv)

kubectl delete secret "${SECRET_NAME}" -n "${NAMESPACE}" 2>/dev/null || true

kubectl create secret docker-registry "${SECRET_NAME}" \
  --docker-server="${ACR_NAME}.azurecr.io" \
  --docker-username="${ACR_NAME}" \
  --docker-password="${ACR_PASSWORD}" \
  --namespace="${NAMESPACE}"

kubectl apply -f namespace.yaml

kubectl apply -f deployment.yaml

kubectl rollout restart deployment/fish-classifier -n "${NAMESPACE}"

kubectl delete pods -n "${NAMESPACE}" -l app=fish-classifier --wait=false 2>/dev/null || true

kubectl rollout status deployment/fish-classifier -n "${NAMESPACE}" --timeout=120s || echo "Rollout may still be in progress"

kubectl apply -f service.yaml
