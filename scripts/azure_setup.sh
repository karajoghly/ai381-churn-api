#!/usr/bin/env bash
# One-time Azure setup for Lab 3. Run it in Azure Cloud Shell (portal.azure.com -> >_ icon, Bash).
#
# Creates: resource group, Container Apps environment, the container app (scale 0..1),
# and a user-assigned managed identity that GitHub Actions logs in as through OIDC:
# GitHub presents a short-lived token for THIS repo's main branch, Azure trusts it,
# and no password or secret ever leaves Azure.
#
#   bash azure_setup.sh                 # default region: italynorth
#   LOCATION=swedencentral bash azure_setup.sh
set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-karajoghly/ai381-churn-api}"
LOCATION="${LOCATION:-italynorth}"
RG="rg-ai381-lab3"
ENV_NAME="cae-ai381-lab3"
APP="churn-api"
IDENTITY="id-github-ai381"
IMAGE="ghcr.io/${GITHUB_REPO,,}:latest"

echo "==> Registering resource providers (first time takes a minute)"
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.ManagedIdentity --wait

# Azure for Students limits which regions you may use. If the subscription has such a
# policy and LOCATION is not allowed, pick an allowed region that runs Container Apps.
ALLOWED=$(az policy assignment list --disable-scope-strict-match \
  --query "[].parameters.listOfAllowedLocations.value[]" -o tsv 2>/dev/null | tr 'A-Z' 'a-z' | sort -u || true)
if [ -n "$ALLOWED" ] && ! grep -qx "$LOCATION" <<<"$ALLOWED"; then
  ACA=$(az provider show -n Microsoft.App \
    --query "resourceTypes[?resourceType=='containerApps'].locations[]" -o tsv \
    | tr 'A-Z' 'a-z' | tr -d ' ')
  for r in $ALLOWED; do
    if grep -qx "$r" <<<"$ACA"; then LOCATION="$r"; break; fi
  done
  echo "    Your subscription only allows: $(echo $ALLOWED) -> using $LOCATION"
fi

echo "==> Resource group $RG in $LOCATION"
az group create -n "$RG" -l "$LOCATION" -o none

echo "==> Container Apps environment (no Log Analytics workspace, to keep cost at zero)"
az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" --logs-destination none -o none

echo "==> Container app $APP (public ingress on port 8000, scale to zero when idle)"
az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" \
  --image "$IMAGE" --target-port 8000 --ingress external \
  --min-replicas 0 --max-replicas 1 --cpu 0.5 --memory 1.0Gi -o none

echo "==> Managed identity + GitHub OIDC trust for $GITHUB_REPO (main branch only)"
az identity create -n "$IDENTITY" -g "$RG" -o none
az identity federated-credential create --name github-main \
  --identity-name "$IDENTITY" -g "$RG" \
  --issuer "https://token.actions.githubusercontent.com" \
  --subject "repo:${GITHUB_REPO}:ref:refs/heads/main" \
  --audiences "api://AzureADTokenExchange" -o none

echo "==> Let the identity manage resources in $RG only (least privilege)"
PRINCIPAL_ID=$(az identity show -n "$IDENTITY" -g "$RG" --query principalId -o tsv)
RG_ID=$(az group show -n "$RG" --query id -o tsv)
az role assignment create --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal --role Contributor --scope "$RG_ID" -o none

FQDN=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
cat <<EOF

================ Add these 3 GitHub repository SECRETS ================
(GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository secret)

AZURE_CLIENT_ID        $(az identity show -n "$IDENTITY" -g "$RG" --query clientId -o tsv)
AZURE_TENANT_ID        $(az account show --query tenantId -o tsv)
AZURE_SUBSCRIPTION_ID  $(az account show --query id -o tsv)

Public URL: https://$FQDN
=======================================================================
EOF
