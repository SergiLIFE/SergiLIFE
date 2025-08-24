@description('Virtual network ID to link')
param vnetId string
@description('List of private DNS zone names to create')
param zones array
@description('Project prefix for naming')
param project string = 'life'
@description('Environment')
param environment string = 'dev'

// Private DNS zones needed for: storage (blob, dfs), key vault, AML workspace endpoints
// Zones passed in by caller (e.g., privatelink.blob.core.windows.net etc.)

resource dnsZones 'Microsoft.Network/privateDnsZones@2020-06-01' = [for z in zones: {
  name: z
  location: 'global'
}]

// Single vnet link per zone
resource vnetLinks 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = [for z in zones: {
  name: '${z}/${project}-${environment}-link'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnetId
    }
  }
}]

output zoneIds array = [for z in zones: resourceId('Microsoft.Network/privateDnsZones', z)]
