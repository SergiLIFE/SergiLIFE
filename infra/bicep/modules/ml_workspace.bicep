@description('AML Workspace name')
param workspaceName string
@description('Location')
param location string = resourceGroup().location
@description('Existing or new Storage Account resource ID')
param storageAccountId string
@description('Key Vault ID (optional for CMK)')
@allowed([ '', 'use' ])
param keyVaultMode string = ''
@description('Key identifier (UNVERSIONED) for CMK encryption (required if keyVaultMode=="use")')
param keyIdentifier string = ''

@description('Enable managed network isolation')
param enableManagedNetwork bool = true

resource ws 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: workspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    description: 'LIFE ML Workspace with ADLS Gen2 and optional CMK'
    storageAccount: storageAccountId
    encryption: keyVaultMode == 'use' ? {
      status: 'Enabled'
      keyVaultProperties: {
        keyVaultArmId: substring(keyIdentifier, 0, lastIndexOf(keyIdentifier, '/keys/'))
        keyIdentifier: keyIdentifier
      }
    } : null
    managedNetwork: enableManagedNetwork ? {
      isolationMode: 'AllowOnlyApprovedOutbound'
    } : null
    publicNetworkAccess: 'Disabled'
    hbiWorkspace: true
  }
}

output workspaceId string = ws.id
output workspacePrincipalId string = ws.identity.principalId
