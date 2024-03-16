variable "aws_region" {
  type        = string
  description = "The AWS region where resources will be created"
  default     = "eu-central-1"
}

variable "aws_ami" {
  type        = string
  description = "The AWS AMI to use for the EC2 instance"
  default     = "ami-02da8ff11275b7907"
}

variable "aws_instance_type" {
  type        = string
  description = "The type of EC2 instance to create"
  default     = "t2.micro"
}
