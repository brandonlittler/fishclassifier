set -e

SUBSCRIPTION_ID="0d688acc-a47f-4540-a274-9e3481ff29ce"
ACR_NAME="fishclassifieracr01"
RESOURCE_GROUP="rg-fish-classifier"
NAMESPACE="fish-classifier"
SECRET_NAME="acr-secret"

az account set --subscription ${SUBSCRIPTION_ID}

az acr update -n ${ACR_NAME} --admin-enabled true

ACR_PASSWORD=$(az acr credential show --name ${ACR_NAME} --resource-group ${RESOURCE_GROUP} --query "passwords[0].value" -o tsv)


kubectl create namespace ${NAMESPACE} 2>/dev/null || echo "Namespace already exists"


kubectl create secret docker-registry ${SECRET_NAME} \
  --docker-server=${ACR_NAME}.azurecr.io \
  --docker-username=${ACR_NAME} \
  --docker-password=${ACR_PASSWORD} \
  --namespace=${NAMESPACE}

kubectl get secret ${SECRET_NAME} -n ${NAMESPACE}

