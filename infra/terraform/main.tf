terraform {
  required_version = ">= 1.6.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 3.100.0"
    }
    random = {
      source = "hashicorp/random"
      version = ">= 3.6.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "location" {
  type    = string
  default = "eastus"
}

variable "prefix" {
  type    = string
  default = "life"
}

variable "container_image" {
  type        = string
  default     = "ghcr.io/OWNER/life-streamlit:latest" # TODO: set to your public GHCR repo
  description = "Container image for the dashboard (must be public or auth configured)."
}

variable "evidence_dir" {
  type    = string
  default = "evidence"
}

resource "azurerm_resource_group" "rg" {
  name     = "${var.prefix}-rg"
  location = var.location
}

resource "random_string" "sfx" {
  length  = 6
  upper   = false
  lower   = true
  numeric = true
  special = false
}

resource "azurerm_log_analytics_workspace" "law" {
  name                = "law${var.prefix}${random_string.sfx.result}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_application_insights" "appi" {
  name                = "appi${var.prefix}${random_string.sfx.result}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.law.id
}

resource "azurerm_storage_account" "st" {
  name                            = "salife${random_string.sfx.result}"
  resource_group_name             = azurerm_resource_group.rg.name
  location                        = azurerm_resource_group.rg.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
}

resource "azurerm_storage_container" "artifacts" {
  name                  = "artifacts"
  storage_account_name  = azurerm_storage_account.st.name
  container_access_type = "private"
}

resource "azurerm_user_assigned_identity" "uami" {
  location            = azurerm_resource_group.rg.location
  name                = "uami-${var.prefix}-${random_string.sfx.result}"
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_key_vault" "kv" {
  name                        = "kvlife${random_string.sfx.result}"
  location                    = azurerm_resource_group.rg.location
  resource_group_name         = azurerm_resource_group.rg.name
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = "standard"
  purge_protection_enabled    = true
  soft_delete_retention_days  = 7
  enable_rbac_authorization   = true
}

data "azurerm_client_config" "current" {}

data "azurerm_role_definition" "kv_secrets_officer" {
  name = "Key Vault Secrets Officer"
  scope = azurerm_key_vault.kv.id
}

resource "azurerm_role_assignment" "uami_kv_secrets_officer" {
  scope              = azurerm_key_vault.kv.id
  role_definition_id = data.azurerm_role_definition.kv_secrets_officer.role_definition_id
  principal_id       = azurerm_user_assigned_identity.uami.principal_id
}

resource "azurerm_service_plan" "asp" {
  name                = "asp-${var.prefix}-${random_string.sfx.result}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  os_type             = "Linux"
  sku_name            = "B1"
}

resource "azurerm_linux_web_app" "app" {
  name                = "app-${var.prefix}-${random_string.sfx.result}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  service_plan_id     = azurerm_service_plan.asp.id

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.uami.id]
  }

  site_config {
    always_on = true
    application_stack {
      docker_image     = split(":", var.container_image)[0]
      docker_image_tag = length(split(":", var.container_image)) > 1 ? split(":", var.container_image)[1] : "latest"
    }
    ftps_state = "Disabled"
    cors {
      allowed_origins = ["*"]
    }
  }

  app_settings = {
    WEBSITES_PORT                         = "8501"
    EVIDENCE_DIR                          = var.evidence_dir
    APPLICATIONINSIGHTS_CONNECTION_STRING = azurerm_application_insights.appi.connection_string
    DOCKER_ENABLE_CI                      = "true"
  }
}

resource "azurerm_monitor_diagnostic_setting" "app_diagnostics" {
  name                       = "diag-${azurerm_linux_web_app.app.name}"
  target_resource_id         = azurerm_linux_web_app.app.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.law.id

  enabled_log {
    category = "AppServiceHTTPLogs"
  }
  enabled_log {
    category = "AppServiceConsoleLogs"
  }
  metric {
    category = "AllMetrics"
    enabled  = true
  }
}

output "resource_group" {
  value = azurerm_resource_group.rg.name
}
output "app_service_name" {
  value = azurerm_linux_web_app.app.name
}
output "app_url" {
  value = "https://${azurerm_linux_web_app.app.default_hostname}"
}
output "storage_account_name" {
  value = azurerm_storage_account.st.name
}
output "key_vault_name" {
  value = azurerm_key_vault.kv.name
}