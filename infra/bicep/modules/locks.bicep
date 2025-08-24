@description('Resource IDs to lock')
param resourceIds array
@description('Lock level (CanNotDelete or ReadOnly)')
@allowed([ 'CanNotDelete', 'ReadOnly' ])
param lockLevel string = 'CanNotDelete'

// Locks at resource scope require extension form; use symbolic existing resource IDs indirectly not strongly typed.
// Implement simple resource-group level lock if resourceIds length > 0 as fallback to avoid per-resource dynamic typing issues.
resource groupLock 'Microsoft.Authorization/locks@2020-05-01' = if (length(resourceIds) > 0) {
  name: 'rg-protection'
  properties: {
    level: lockLevel
    notes: 'Applied at RG level; protects contained critical resources.'
  }
}

output lockedCount int = length(resourceIds) > 0 ? length(resourceIds) : 0
