@description('Location for all resources')
param location string = resourceGroup().location

@description('Environment suffix for naming (e.g., dev, prod)')
param environment string = 'dev'

@description('Project name for resource naming')
param projectName string = 'life'

@description('Event Hub capacity (1-10)')
@minValue(1)
@maxValue(10)
param eventHubCapacity int = 1

@description('Application Insights workspace SKU')
@allowed(['PerGB2018', 'Free', 'Standalone', 'PerNode', 'Standard', 'Premium'])
param workspaceSku string = 'PerGB2018'

// Generate unique resource names
var resourceSuffix = '${projectName}-${environment}-${uniqueString(resourceGroup().id)}'
var eventHubNamespaceName = 'evhns-${resourceSuffix}'
var eventHubName = 'intervention-events'
var applicationInsightsName = 'appi-${resourceSuffix}'
var logAnalyticsWorkspaceName = 'law-${resourceSuffix}'

// Event Hub Namespace
resource eventHubNamespace 'Microsoft.EventHub/namespaces@2023-01-01-preview' = {
  name: eventHubNamespaceName
  location: location
  sku: {
    name: 'Standard'
    tier: 'Standard'
    capacity: eventHubCapacity
  }
  properties: {
    minimumTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
    zoneRedundant: false
    isAutoInflateEnabled: false
    maximumThroughputUnits: 0
    kafkaEnabled: true
  }
}

// Event Hub for intervention events
resource eventHub 'Microsoft.EventHub/namespaces/eventhubs@2023-01-01-preview' = {
  parent: eventHubNamespace
  name: eventHubName
  properties: {
    messageRetentionInDays: 1
    partitionCount: 2
    status: 'Active'
  }
}

// Event Hub Authorization Rule for sending events
resource eventHubAuthRule 'Microsoft.EventHub/namespaces/eventhubs/authorizationRules@2023-01-01-preview' = {
  parent: eventHub
  name: 'SendRule'
  properties: {
    rights: ['Send']
  }
}

// Event Hub Authorization Rule for consuming events
resource eventHubListenRule 'Microsoft.EventHub/namespaces/eventhubs/authorizationRules@2023-01-01-preview' = {
  parent: eventHub
  name: 'ListenRule'
  properties: {
    rights: ['Listen']
  }
}

// Log Analytics Workspace for Application Insights
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  properties: {
    sku: {
      name: workspaceSku
    }
    retentionInDays: 30
    features: {
      searchVersion: 1
      legacy: 0
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

// Application Insights
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    Flow_Type: 'Redfield'
    Request_Source: 'rest'
    RetentionInDays: 30
    WorkspaceResourceId: logAnalyticsWorkspace.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// Outputs for application configuration
@description('Event Hub Namespace name')
output eventHubNamespaceName string = eventHubNamespace.name

@description('Event Hub name')
output eventHubName string = eventHub.name

@description('Event Hub Namespace FQDN')
output eventHubNamespaceFqdn string = eventHubNamespace.properties.serviceBusEndpoint

@description('Application Insights name')
output applicationInsightsName string = applicationInsights.name

@description('Application Insights connection string')
@secure()
output applicationInsightsConnectionString string = applicationInsights.properties.ConnectionString

@description('Log Analytics Workspace name')
output logAnalyticsWorkspaceName string = logAnalyticsWorkspace.name

@description('Log Analytics Workspace ID')
output logAnalyticsWorkspaceId string = logAnalyticsWorkspace.id

@description('Resource group name for reference')
output resourceGroupName string = resourceGroup().name
