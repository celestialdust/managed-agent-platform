# Four IMMUTABLE platform repositories, then one MUTABLE spike repository. The
# difference is declared rather than remembered: core/environment.py refuses any
# image reference that is not name@sha256:<64 hex>, and MAP-56's D4 rests that rule
# on IMMUTABLE tags. A repository quietly flipped to MUTABLE would leave the rule
# standing on nothing.
import {
  to = aws_ecr_repository.control_plane
  id = "map/control-plane"
}
resource "aws_ecr_repository" "control_plane" {
  name                 = "map/control-plane"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
}

import {
  to = aws_ecr_repository.model_gateway
  id = "map/model-gateway"
}
resource "aws_ecr_repository" "model_gateway" {
  name                 = "map/model-gateway"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
}

import {
  to = aws_ecr_repository.session_shim
  id = "map/session-shim"
}
resource "aws_ecr_repository" "session_shim" {
  name                 = "map/session-shim"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
}

import {
  to = aws_ecr_repository.tool_gateway
  id = "map/tool-gateway"
}
resource "aws_ecr_repository" "tool_gateway" {
  name                 = "map/tool-gateway"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
}

# MAP-1's, and MUTABLE on purpose -- a spike image is rebuilt under one name.
import {
  to = aws_ecr_repository.spike
  id = "map/spike"
}
resource "aws_ecr_repository" "spike" {
  name                 = "map/spike"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
}
