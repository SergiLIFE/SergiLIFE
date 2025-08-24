@description('Base name / project prefix')
param project string = 'life'
@description('Environment tag')
param environment string = 'dev'
@description('Location')
param location string = resourceGroup().location

@description('Virtual Network name')
param vnetName string = '${project}-${environment}-vnet'

@description('Storage account name (lowercase, 3-24 chars)')
param storageAccountName string

@description('Key Vault name (3-24 chars)')
param keyVaultName string

@description('AML Workspace name')
param workspaceName string = '${project}-${environment}-mlw'

@description('Create and use customer managed key')
param useCmk bool = true

@description('Admin object IDs for Key Vault access policies')
param adminObjectIds array = []

@description('Enable resource locks')
param enableLocks bool = true
@description('Log Analytics retention days')
param logAnalyticsRetentionDays int = 90
@description('Lock level for protected resources')
@allowed([ 'CanNotDelete', 'ReadOnly' ])
param lockLevel string = 'CanNotDelete'
@description('Function App name for key rotation')
param keyRotationFunctionName string = '${project}-${environment}-kvrotate'
@description('Function storage account name')
param keyRotationStorageName string = replace('${project}${environment}kvrotstg', '-', '')
@description('Storage DNS suffix used to build private DNS zones (provide appropriate value per cloud)')
param storageDnsSuffix string

var suffix = uniqueString(resourceGroup().id, project, environment)

module network 'modules/network_shared.bicep' = {
  name: 'network'
  params: {
    vnetName: vnetName
    location: location
  }
}

// Always deploy key vault (even if CMK not used) for future extensibility
module keyvault 'modules/keyvault_cmk.bicep' = {
  name: 'keyvault'
  params: {
    keyVaultName: keyVaultName
    location: location
    adminObjectIds: adminObjectIds
  }
}

module storage 'modules/storage_adls.bicep' = {
  name: 'storage'
  params: {
    storageAccountName: storageAccountName
    location: location
  }
}

// Single workspace module with optional encryption parameters
var encryptionMode = useCmk ? 'use' : ''
var encryptionKeyId = useCmk ? keyvault.outputs.unversionedKeyId : ''

module workspace 'modules/ml_workspace.bicep' = {
  name: 'mlworkspace'
  params: {
    workspaceName: workspaceName
    location: location
    storageAccountId: storage.outputs.storageAccountId
    keyVaultMode: encryptionMode
  keyIdentifier: encryptionKeyId
  }
}

module pe 'modules/private_endpoints.bicep' = {
  name: 'privateEndpoints'
  params: {
    subnetId: network.outputs.privateEndpointSubnetId
    storageAccountId: storage.outputs.storageAccountId
    workspaceId: workspace.outputs.workspaceId
    suffix: suffix
    location: location
  }
}

// Log Analytics (diagnostics sink) - always deployed
module logAnalytics 'modules/log_analytics.bicep' = {
  name: 'logAnalytics'
  params: {
    workspaceName: '${project}-${environment}-law'
    location: location
    retentionDays: logAnalyticsRetentionDays
  }
}

// Key rotation function - always deployed
module keyRotate 'modules/function_keyrotation.bicep' = {
  name: 'keyRotation'
  params: {
    functionAppName: keyRotationFunctionName
    funcStorageName: keyRotationStorageName
    keyVaultUri: keyvault.outputs.keyVaultId
  }
}

// Event Grid subscription for key near expiry -> function - always deployed
module keyEvents 'modules/eventgrid_subscription.bicep' = {
  name: 'keyEvents'
  params: {
    keyVaultName: keyVaultName
    functionResourceId: keyRotate.outputs.functionResourceId
  }
}

// Private DNS zones and links
module privateDns 'modules/private_dns.bicep' = {
  name: 'privateDns'
  params: {
    vnetId: network.outputs.vnetId
    project: project
    environment: environment
    zones: [
      'privatelink.blob.${storageDnsSuffix}'
      'privatelink.dfs.${storageDnsSuffix}'
      'privatelink.vaultcore.azure.net'
      'privatelink.api.azureml.ms'
    ]
  }
}

// Locks on critical resources (using existing resource references)
resource kvExisting 'Microsoft.KeyVault/vaults@2023-07-01' existing = if (enableLocks) {
  name: keyVaultName
}
resource storageAcct 'Microsoft.Storage/storageAccounts@2023-05-01' existing = if (enableLocks) {
  name: storageAccountName
}
resource amlWs 'Microsoft.MachineLearningServices/workspaces@2024-04-01' existing = if (enableLocks) {
  name: workspaceName
}

resource lockKv 'Microsoft.Authorization/locks@2020-05-01' = if (enableLocks) {
  name: 'lock-kv'
  scope: kvExisting
  properties: {
    level: lockLevel
    notes: 'Protected Key Vault'
  }
}
resource lockStg 'Microsoft.Authorization/locks@2020-05-01' = if (enableLocks) {
  name: 'lock-stg'
  scope: storageAcct
  properties: {
    level: lockLevel
    notes: 'Protected Storage Account'
  }
}
resource lockMlw 'Microsoft.Authorization/locks@2020-05-01' = if (enableLocks) {
  name: 'lock-mlw'
  scope: amlWs
  properties: {
    level: lockLevel
    notes: 'Protected AML Workspace'
  }
}

output workspaceId string = workspace.outputs.workspaceId
output storageAccountId string = storage.outputs.storageAccountId
// Safe keyId output (empty when CMK not used)
output keyId string = encryptionKeyId
output vnetId string = network.outputs.vnetId
output privateEndpointSubnetId string = network.outputs.privateEndpointSubnetId
output storagePrivateEndpointId string = pe.outputs.storagePrivateEndpointId
output mlPrivateEndpointId string = pe.outputs.mlPrivateEndpointId
// Direct outputs (modules always deployed)
output logAnalyticsWorkspaceId string = logAnalytics.outputs.logAnalyticsWorkspaceId
output keyRotationFunctionId string = keyRotate.outputs.functionResourceId
output keyRotationEventSubscriptionId string = keyEvents.outputs.eventSubscriptionId
output locksApplied int = enableLocks ? 3 : 0

// Diagnostics settings module (after core resources & function) using unversioned key workspace id
module diagnostics 'modules/diagnostics.bicep' = {
  name: 'diagnostics'
  params: {
    logAnalyticsWorkspaceId: logAnalytics.outputs.logAnalyticsWorkspaceId
    storageAccountId: storage.outputs.storageAccountId
    keyVaultId: keyvault.outputs.keyVaultId
    workspaceId: workspace.outputs.workspaceId
    functionAppId: keyRotate.outputs.functionResourceId
  }
}

// RBAC Role Assignments for Function to access Key Vault (Key Vault Crypto User + Reader) and Storage data if needed
@description('Built-in role definition IDs')
var roleKeyVaultCryptoUser = '14b46e9e-c2b7-41b4-b07b-48a6ebf60603' // Key Vault Crypto User
var roleKeyVaultReader = '21090545-7ca7-4776-b22c-e363652d74d2' // Key Vault Reader

resource kvForRbac 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource rbacKvCrypto 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVaultName, keyRotationFunctionName, roleKeyVaultCryptoUser)
  scope: kvForRbac
  properties: {
    principalId: keyRotate.outputs.functionPrincipalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleKeyVaultCryptoUser)
  }
}
resource rbacKvReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVaultName, keyRotationFunctionName, roleKeyVaultReader)
  scope: kvForRbac
  properties: {
    principalId: keyRotate.outputs.functionPrincipalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleKeyVaultReader)
  }
}
