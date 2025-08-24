@description('Function App name')
param functionAppName string
@description('Location')
param location string = resourceGroup().location
@description('Storage account name for Functions (lowercase)')
param funcStorageName string
@description('Key Vault URI')
param keyVaultUri string
@description('Runtime version')
param functionsExtensionVersion string = '~4'

// Retrieve storage key via resource instance function and build connection string
var storageKey = funcstg.listKeys().keys[0].value
var azureWebJobsStorage = 'DefaultEndpointsProtocol=https;AccountName=${funcStorageName};AccountKey=${storageKey};EndpointSuffix=${environment().suffixes.storage}'

resource funcstg 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: funcStorageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
  }
}

resource plan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: '${functionAppName}-plan'
  location: location
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
  }
}

resource func 'Microsoft.Web/sites@2023-12-01' = {
  name: functionAppName
  location: location
  kind: 'functionapp'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: plan.id
    httpsOnly: true
    siteConfig: {
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: azureWebJobsStorage
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: functionsExtensionVersion
        }
        {
          name: 'KEY_VAULT_URI'
          value: keyVaultUri
        }
      ]
    }
  }
}

output functionPrincipalId string = func.identity.principalId
output functionResourceId string = func.id
output defaultHostname string = func.properties.defaultHostName
