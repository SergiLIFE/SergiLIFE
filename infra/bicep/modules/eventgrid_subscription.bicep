@description('Key Vault name')
param keyVaultName string
@description('Azure Function resource ID to invoke')
param functionResourceId string
@description('Event subscription name')
param eventSubscriptionName string = 'kv-keynear-expiry'

// Existing Key Vault in same resource group
resource kv 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource kvEventSub 'Microsoft.EventGrid/eventSubscriptions@2023-06-01-preview' = {
  name: eventSubscriptionName
  scope: kv
  properties: {
    destination: {
      endpointType: 'AzureFunction'
      properties: {
        resourceId: functionResourceId
      }
    }
    filter: {
      includedEventTypes: [ 'Microsoft.KeyVault.KeyNearExpiry' ]
      isSubjectCaseSensitive: false
    }
    eventDeliverySchema: 'EventGridSchema'
  }
}

output eventSubscriptionId string = kvEventSub.id
