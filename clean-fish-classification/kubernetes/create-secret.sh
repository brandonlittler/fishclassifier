set -e

# Set these in your environment (or .env) - do not commit real values
SUBSCRIPTION_ID="${SUBSCRIPTION_ID:?Set SUBSCRIPTION_ID}"
ACR_NAME="${ACR_NAME:?Set ACR_NAME}"
RESOURCE_GROUP="${RESOURCE_GROUP:?Set RESOURCE_GROUP}"
NAMESPACE="${NAMESPACE:-fish-classifier}"
SECRET_NAME="${SECRET_NAME:-acr-secret}"

az account set --subscription "${SUBSCRIPTION_ID}"

az acr update -n "${ACR_NAME}" --admin-enabled true

ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --resource-group "${RESOURCE_GROUP}" --query "passwords[0].value" -o tsv)


kubectl create namespace "${NAMESPACE}" 2>/dev/null || echo "Namespace already exists"


kubectl create secret docker-registry "${SECRET_NAME}" \
  --docker-server="${ACR_NAME}.azurecr.io" \
  --docker-username="${ACR_NAME}" \
  --docker-password="${ACR_PASSWORD}" \
  --namespace="${NAMESPACE}"

kubectl get secret "${SECRET_NAME}" -n "${NAMESPACE}"

