@description('Key Vault name')
param keyVaultName string
@description('Location')
param location string = resourceGroup().location
@description('SKU')
@allowed(['standard', 'premium'])
param skuName string = 'standard'
@description('Administrator object IDs allowed to manage keys/secrets (array)')
param adminObjectIds array = []

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    enableSoftDelete: true
    enableRbacAuthorization: false
    sku: {
      family: 'A'
      name: skuName
    }
    accessPolicies: [for id in adminObjectIds: {
      tenantId: subscription().tenantId
      objectId: id
      permissions: {
        keys: [ 'Get', 'List', 'Create', 'Update', 'WrapKey', 'UnwrapKey' ]
        secrets: [ 'Get', 'List', 'Set' ]
        certificates: [ 'Get', 'List' ]
      }
    }]
    enabledForDeployment: false
    enabledForTemplateDeployment: true
    enabledForDiskEncryption: false
    publicNetworkAccess: 'Enabled'
  }
}

@description('Name of the CMK key to create')
param keyName string = 'ml-cmk'

resource cmk 'Microsoft.KeyVault/vaults/keys@2023-07-01' = {
  name: keyName
  parent: kv
  properties: {
    kty: 'RSA'
    keySize: 2048
  }
}

output keyVaultId string = kv.id
// Versioned key id
output keyId string = cmk.id
// Unversioned key identifier for services that should auto-rotate
output unversionedKeyId string = substring(cmk.id, 0, lastIndexOf(cmk.id, '/'))
