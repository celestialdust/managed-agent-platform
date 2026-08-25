# 5432 from the two private CIDRs and nowhere else. This is the rule that keeps
# the Event Log off the internet, and the CIDRs are read from the subnet resources
# rather than typed again, so widening a subnet cannot silently leave this behind.
import {
  to = aws_security_group.db
  id = "sg-0ab889ef74200bae7"
}
resource "aws_security_group" "db" {
  name        = "map-dev-db"
  description = "map-dev-db: Postgres from the private subnets only"
  vpc_id      = aws_default_vpc.map.id

  ingress {
    protocol  = "tcp"
    from_port = 5432
    to_port   = 5432
    cidr_blocks = [
      aws_subnet.private_1a.cidr_block,
      aws_subnet.private_1b.cidr_block,
    ]
  }

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

import {
  to = aws_db_subnet_group.map_dev
  id = "map-dev-db"
}
resource "aws_db_subnet_group" "map_dev" {
  name        = "map-dev-db"
  description = "map-dev private subnets"
  subnet_ids  = [aws_subnet.private_1a.id, aws_subnet.private_1b.id]
}

# THERE IS NO PASSWORD HERE, AND THAT IS NOT AN OMISSION.
#
# RDS manages and rotates the master password: describe-db-instances reports a
# MasterUserSecret whose SecretArn names the AWS-owned `rds!db-...` secret. So the
# only thing Terraform ever reads is that ARN. Measured on the plan JSON: the four
# keys `manage_master_user_password`, `password`, `password_wo` and
# `password_wo_version` are all null, and `secret_string` appears nowhere.
#
# `manage_master_user_password = true` is deliberately NOT written. The provider
# does not refresh it from the API, so it can never detect the drift it appears to
# guard -- it would be a declaration nobody checks, which is the exact defect
# ADR-021 exists to remove. Writing it also costs a permanent-looking diff on the
# first plan and buys nothing.
#
# `instance_class` is db.t4g.medium. It has now been resized twice for the same
# reason -- db.t4g.micro -> db.t4g.small on 2026-08-22, and small -> medium on
# 2026-08-23 -- and both times because max_connections ran out, not because CPU or
# storage did. That first resize is the reason this file exists: nothing in the
# repository could have recorded it.
#
# The second one has a number behind it. On db.t4g.small, measured from inside a
# control-plane pod: `show max_connections` = 181, three reserved for superusers,
# so 178 for map_app. The platform's own declared demand was 180 -- one
# control-plane and two tool-gateway processes at pool_size 50 + max_overflow 10 --
# so it was over its database before a single Session ran. It had passed a guard
# that compared against 225, a figure computed from the parameter group's formula
# and never measured, and 47 too high.
#
# medium was chosen to fit ADR-029's plan of 50 concurrent Sessions
# rather than to clear the current number by a margin. Dropping pool_size to fit
# the small instance was the alternative and was rejected: it caps the plan at 30
# and tests/test_composition.py records a MEASURED 6x cliff behind that
# parameter -- 50 concurrent appends took 20.5 ms at size=50 and 132 ms at size=40.
#
# `default.postgres17` is an AWS default parameter group and cannot be a managed
# resource, so it is named by string -- which still makes a change OF parameter
# group show up as drift here. max_connections comes from that group's formula,
# `LEAST({DBInstanceClassMemory/9531392}, 5000)`, so the instance class IS the
# connection ceiling and there is no knob here to turn instead.
import {
  to = aws_db_instance.map_dev
  id = "map-dev-db"
}
resource "aws_db_instance" "map_dev" {
  identifier     = "map-dev-db"
  engine         = "postgres"
  engine_version = "17.10"
  instance_class = "db.t4g.medium"

  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true
  kms_key_id        = "arn:aws:kms:us-east-1:${local.account_id}:key/244d6552-ae49-4f43-8655-0ae9d9e9cbda"

  db_name  = "managed_agent"
  username = "mapadmin"
  port     = 5432

  db_subnet_group_name   = aws_db_subnet_group.map_dev.name
  vpc_security_group_ids = [aws_security_group.db.id]
  parameter_group_name   = "default.postgres17"
  publicly_accessible    = false
  multi_az               = false
  availability_zone      = "us-east-1b"

  backup_retention_period = 7
  backup_window           = "04:54-05:24"
  maintenance_window      = "fri:04:07-fri:04:37"

  # Without this a change to instance_class is queued for the maintenance window above --
  # Friday 04:07 -- and terraform reports success having changed nothing anyone can
  # observe until then. That is the worst shape a deploy can take: the plan is applied,
  # the gate is green, and the ceiling the change exists to raise is still the old one for
  # days. It is `true` because this is the development account and a resize here costs a
  # reboot of about a minute against no tenant; a production account would want the
  # opposite and would say so in its own tfvars.
  apply_immediately          = true
  auto_minor_version_upgrade = true
  ca_cert_identifier         = "rds-ca-rsa2048-g1"
  copy_tags_to_snapshot      = false
  deletion_protection        = false

  tags = { Name = "map-dev-db" }

  lifecycle {
    prevent_destroy = true
  }
}
