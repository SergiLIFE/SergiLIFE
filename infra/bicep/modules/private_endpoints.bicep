@description('Location')
param location string = resourceGroup().location
@description('Subnet resource ID for private endpoints')
param subnetId string
@description('Storage account resource ID')
param storageAccountId string
@description('Workspace resource ID')
param workspaceId string
@description('Unique suffix for naming')
param suffix string

resource pepStorage 'Microsoft.Network/privateEndpoints@2023-09-01' = {
  name: 'pep-stg-${suffix}'
  location: location
  properties: {
    subnet: { id: subnetId }
    privateLinkServiceConnections: [
      {
        name: 'stgConnection'
        properties: {
          groupIds: [ 'blob', 'file', 'dfs' ]
          privateLinkServiceId: storageAccountId
          requestMessage: 'Access for LIFE workspace storage'
        }
      }
    ]
  }
}

resource pepMl 'Microsoft.Network/privateEndpoints@2023-09-01' = {
  name: 'pep-ml-${suffix}'
  location: location
  properties: {
    subnet: { id: subnetId }
    privateLinkServiceConnections: [
      {
        name: 'mlConnection'
        properties: {
          groupIds: [ 'amlworkspace' ]
          privateLinkServiceId: workspaceId
          requestMessage: 'Access for LIFE AML workspace'
        }
      }
    ]
  }
}

output storagePrivateEndpointId string = pepStorage.id
output mlPrivateEndpointId string = pepMl.id
