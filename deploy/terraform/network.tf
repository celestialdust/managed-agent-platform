# The VPC is the account's DEFAULT VPC, not one built for this platform:
# describe-vpcs reports IsDefault true and 172.31.0.0/16, with no tags at all.
# `aws_default_vpc` is the resource type for adopting that -- it takes no
# arguments, reads the VPC as it is, and does not claim to have created it. An
# empty block imports with zero diff; measured.
import {
  to = aws_default_vpc.map
  id = "vpc-05abe06a50851a1b1"
}
resource "aws_default_vpc" "map" {
  lifecycle {
    prevent_destroy = true
  }
}

# A default subnet, and it is load-bearing twice: the NAT gateway lives in it, so
# every node's egress runs through it, and it is one of the cluster's four
# subnetIds. Four more default subnets exist (1c, 1d, 1e, 1f) and are deliberately
# not declared -- nothing references them, so nothing about this platform can
# change when they do.
import {
  to = aws_default_subnet.public_1a
  id = "subnet-03963d5a0da408991"
}
resource "aws_default_subnet" "public_1a" {
  availability_zone = "us-east-1a"
  tags              = { "kubernetes.io/role/elb" = "1" }
  lifecycle {
    prevent_destroy = true
  }
}

import {
  to = aws_default_subnet.public_1b
  id = "subnet-002224bef2c851f86"
}
resource "aws_default_subnet" "public_1b" {
  availability_zone = "us-east-1b"
  tags              = { "kubernetes.io/role/elb" = "1" }
  lifecycle {
    prevent_destroy = true
  }
}

# The two subnets somebody actually made, and the two this platform actually runs
# in: the nodegroup places every node in them and `aws_db_subnet_group.map_dev`
# pins RDS to them.
#
# An earlier version of this file left them unguarded, reasoning that "losing one
# costs a rebuild and an RDS move, not data". That is the wrong way round. AWS
# will hand you another *default* subnet on request; nothing hands you these back,
# and a rebuild plus an RDS move is exactly the class of event prevent_destroy
# exists to refuse. It costs one thing, stated so nobody has to rediscover it: a
# deliberate re-CIDR of a private subnet is now exit 1 and a refusal rather than a
# diff, and the way through is to remove the guard in the same commit that does it.
import {
  to = aws_subnet.private_1a
  id = "subnet-0badee1628fa8f826"
}
resource "aws_subnet" "private_1a" {
  vpc_id                  = aws_default_vpc.map.id
  availability_zone       = "us-east-1a"
  cidr_block              = "172.31.96.0/20"
  map_public_ip_on_launch = false
  tags = {
    Name                              = "map-dev-private-1a"
    "kubernetes.io/role/internal-elb" = "1"
  }
  lifecycle {
    prevent_destroy = true
  }
}

import {
  to = aws_subnet.private_1b
  id = "subnet-08539b97f50270799"
}
resource "aws_subnet" "private_1b" {
  vpc_id                  = aws_default_vpc.map.id
  availability_zone       = "us-east-1b"
  cidr_block              = "172.31.112.0/20"
  map_public_ip_on_launch = false
  tags = {
    Name                              = "map-dev-private-1b"
    "kubernetes.io/role/internal-elb" = "1"
  }
  lifecycle {
    prevent_destroy = true
  }
}

# The default VPC's internet gateway. The NAT gateway's own egress goes through
# it, so detaching it takes the entire cluster off the internet -- nodes stop
# joining, images stop pulling. There is no `aws_default_internet_gateway`, so it
# is imported as `aws_internet_gateway`, which means Terraform now believes it
# created it. prevent_destroy is what contains that.
import {
  to = aws_internet_gateway.default
  id = "igw-0285252a8503a66b4"
}
resource "aws_internet_gateway" "default" {
  vpc_id = aws_default_vpc.map.id
  lifecycle {
    prevent_destroy = true
  }
}

# The VPC's MAIN route table, holding the one 0.0.0.0/0 -> igw route the NAT
# needs. `aws_default_route_table` is the right type and does not work: imported
# alone in an empty directory it fails `Error: empty result`, exit 1, printing
# nothing else (provider 6.61.0). As `aws_route_table` it imports with zero diff.
# The cost of the substitution is that a destroy would delete the default VPC's
# main route table; prevent_destroy turns that into exit 1 and a refusal.
import {
  to = aws_route_table.main
  id = "rtb-0d9090b23d980f710"
}
resource "aws_route_table" "main" {
  vpc_id = aws_default_vpc.map.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.default.id
  }
  lifecycle {
    prevent_destroy = true
  }
}

# 3.91.142.166 -- the address every outbound connection from every Session pod
# appears to come from. Releasing it means an upstream allowlist somewhere stops
# matching, which is not a failure this platform can see.
import {
  to = aws_eip.nat
  id = "eipalloc-088d31e9cc16cb635"
}
resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "map-dev-nat" }
  lifecycle {
    prevent_destroy = true
  }
}

import {
  to = aws_nat_gateway.map
  id = "nat-0d4a9d1c29d3baa4b"
}
resource "aws_nat_gateway" "map" {
  allocation_id = aws_eip.nat.allocation_id
  subnet_id     = aws_default_subnet.public_1a.id
  tags          = { Name = "map-dev-nat" }
  lifecycle {
    prevent_destroy = true
  }
}

# The route table's `local` route is not declared and must not be: the provider
# filters the implicit VPC-local route out of what it reads, so declaring it would
# be a permanent diff.
import {
  to = aws_route_table.private
  id = "rtb-0f1793f4f31bf2d84"
}
resource "aws_route_table" "private" {
  vpc_id = aws_default_vpc.map.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.map.id
  }
  tags = { Name = "map-dev-private" }
}

# An association is its own resource with its own id shape, `subnet/table`. Left
# out, the route table above would be declared and attached to nothing, and a
# detachment -- which is what would actually take the nodes offline -- would not
# be drift.
import {
  to = aws_route_table_association.private_1a
  id = "subnet-0badee1628fa8f826/rtb-0f1793f4f31bf2d84"
}
resource "aws_route_table_association" "private_1a" {
  subnet_id      = aws_subnet.private_1a.id
  route_table_id = aws_route_table.private.id
}

import {
  to = aws_route_table_association.private_1b
  id = "subnet-08539b97f50270799/rtb-0f1793f4f31bf2d84"
}
resource "aws_route_table_association" "private_1b" {
  subnet_id      = aws_subnet.private_1b.id
  route_table_id = aws_route_table.private.id
}
