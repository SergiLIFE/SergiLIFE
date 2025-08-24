@description('Storage account name (must be globally unique, lowercase)')
param storageAccountName string
@description('Location')
param location string = resourceGroup().location
@description('SKU')
param skuName string = 'Standard_LRS'
@description('Kind')
param kind string = 'StorageV2'
@description('Enable hierarchical namespace for ADLS Gen2')
param enableHns bool = true

resource stg 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: skuName
  }
  kind: kind
  properties: {
    isHnsEnabled: enableHns
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
      ipRules: []
      virtualNetworkRules: []
    }
    encryption: {
      services: {
        blob: { enabled: true }
        file: { enabled: true }
      }
      keySource: 'Microsoft.Storage'
    }
    publicNetworkAccess: 'Enabled'
  }
}

@description('Containers to create')
param containerNames array = [ 'life-data', 'life-raw', 'life-output' ]

resource containers 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = [for c in containerNames: {
  name: '${storageAccountName}/default/${c}'
  properties: {
    publicAccess: 'None'
  }
}]

output storageAccountId string = stg.id
output primaryEndpoints object = stg.properties.primaryEndpoints
