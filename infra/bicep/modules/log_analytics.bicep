@description('Log Analytics workspace name')
param workspaceName string
@description('Location')
param location string = resourceGroup().location
@description('Retention in days')
param retentionDays int = 90

resource law 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: workspaceName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: retentionDays
    features: {
      searchVersion: 1
    }
  }
}

output logAnalyticsWorkspaceId string = law.id
