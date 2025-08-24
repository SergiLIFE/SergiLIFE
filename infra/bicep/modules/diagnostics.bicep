@description('Log Analytics workspace ID')
param logAnalyticsWorkspaceId string
@description('Apply diagnostics for storage account')
param storageAccountId string = ''
@description('Apply diagnostics for key vault')
param keyVaultId string = ''
@description('Apply diagnostics for AML workspace')
param workspaceId string = ''
@description('Apply diagnostics for function app')
param functionAppId string = ''

// Existing resources (conditionally) for typed scope
resource stg 'Microsoft.Storage/storageAccounts@2023-05-01' existing = if (!empty(storageAccountId)) {
  name: last(split(storageAccountId, '/'))
}
resource kv 'Microsoft.KeyVault/vaults@2023-07-01' existing = if (!empty(keyVaultId)) {
  name: last(split(keyVaultId, '/'))
}
resource mlw 'Microsoft.MachineLearningServices/workspaces@2024-04-01' existing = if (!empty(workspaceId)) {
  name: last(split(workspaceId, '/'))
}
resource func 'Microsoft.Web/sites@2023-12-01' existing = if (!empty(functionAppId)) {
  name: last(split(functionAppId, '/'))
}

resource diagStg 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(storageAccountId)) {
  name: 'diag-stg'
  scope: stg
  properties: {
    workspaceId: logAnalyticsWorkspaceId
    logs: [ { categoryGroup: 'allLogs', enabled: true } ]
    metrics: [ { category: 'AllMetrics', enabled: true } ]
  }
}
resource diagKv 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(keyVaultId)) {
  name: 'diag-kv'
  scope: kv
  properties: {
    workspaceId: logAnalyticsWorkspaceId
    logs: [ { categoryGroup: 'allLogs', enabled: true } ]
    metrics: [ { category: 'AllMetrics', enabled: true } ]
  }
}
resource diagMlw 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(workspaceId)) {
  name: 'diag-mlw'
  scope: mlw
  properties: {
    workspaceId: logAnalyticsWorkspaceId
    logs: [ { categoryGroup: 'allLogs', enabled: true } ]
    metrics: [ { category: 'AllMetrics', enabled: true } ]
  }
}
resource diagFunc 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(functionAppId)) {
  name: 'diag-func'
  scope: func
  properties: {
    workspaceId: logAnalyticsWorkspaceId
    logs: [ { categoryGroup: 'allLogs', enabled: true } ]
    metrics: [ { category: 'AllMetrics', enabled: true } ]
  }
}

var countApplied = (empty(storageAccountId) ? 0 : 1) + (empty(keyVaultId) ? 0 : 1) + (empty(workspaceId) ? 0 : 1) + (empty(functionAppId) ? 0 : 1)

output diagnosticsApplied int = countApplied
