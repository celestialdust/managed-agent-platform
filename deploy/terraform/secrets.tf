# Two secret CONTAINERS. Their VERSIONS, which is where the values are, are
# deliberately absent -- Terraform state records every attribute it reads, so an
# `aws_secretsmanager_secret_version` import writes a live credential into the state
# file. `test_no_declared_resource_can_hold_a_secret_value` refuses the resource
# type outright rather than trusting anyone to remember this.
#
# The third secret in the account, `rds!db-21d8250a-...`, is not here and cannot be:
# `OwningService: rds`, `RotationEnabled: true`, and the resource's own name
# validator rejects the `!` -- `Error: only alphanumeric characters and /_+=.@-
# special characters are allowed in "name"`. AWS creates, rotates and recreates it.
import {
  to = aws_secretsmanager_secret.platform_db
  id = "arn:aws:secretsmanager:us-east-1:${local.account_id}:secret:map/dev/platform/db-qOB2YZ"
}
resource "aws_secretsmanager_secret" "platform_db" {
  name        = "map/dev/platform/db"
  description = "map-dev-db connection details. No password here: it lives in the RDS-managed secret named by password_secret_arn, which AWS rotates."
  lifecycle {
    prevent_destroy = true
  }
}

import {
  to = aws_secretsmanager_secret.provider_anthropic
  id = "arn:aws:secretsmanager:us-east-1:${local.account_id}:secret:map/dev/providers/anthropic-yNBhsj"
}
resource "aws_secretsmanager_secret" "provider_anthropic" {
  name        = "map/dev/providers/anthropic"
  description = "Anthropic models via Azure Foundry: api_key, base_url, model"
  lifecycle {
    prevent_destroy = true
  }
}
