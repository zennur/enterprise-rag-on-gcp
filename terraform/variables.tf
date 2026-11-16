variable "project_id" {
  type        = string
  description = "GCP project id. The stack is not applied by CI."
}

variable "region" {
  type        = string
  description = "Region for Cloud Run, Artifact Registry, and Pub/Sub. Match GCP_LOCATION."
  default     = "us-central1"
}

variable "name_prefix" {
  type        = string
  description = "Short prefix for resource names."
  default     = "rag"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,12}$", var.name_prefix))
    error_message = "name_prefix must be lowercase and short enough for service account ids."
  }
}

variable "image" {
  type        = string
  description = "Image to deploy. Defaults to the Artifact Registry path this stack creates. Push it before apply."
  default     = ""
}

variable "vector_store" {
  type        = string
  description = "memory is a smoke test: instances do not share an index. Use pgvector or vertex for a real corpus."
  default     = "memory"

  validation {
    condition     = contains(["memory", "pgvector", "vertex"], var.vector_store)
    error_message = "vector_store must be memory, pgvector, or vertex."
  }
}

variable "vertex_catalog_database" {
  type        = bool
  description = "When vector_store is vertex, read chunk text from DATABASE_URL instead of process memory."
  default     = false
}

variable "vector_search_index" {
  type        = string
  description = "Full resource name of a streaming Vector Search index. Not created here."
  default     = ""
}

variable "vector_search_endpoint" {
  type        = string
  description = "Full resource name of the index endpoint. Not created here."
  default     = ""
}

variable "vector_search_deployed_index_id" {
  type    = string
  default = ""
}

variable "embedding_model" {
  type    = string
  default = "gemini-embedding-001"
}

variable "embedding_dimensionality" {
  type    = number
  default = 768
}

variable "generation_model" {
  type    = string
  default = "gemini-2.5-flash"
}

variable "api_allow_unauthenticated" {
  type        = bool
  description = "Grant allUsers run.invoker on the API. Leave false and put IAP or a load balancer in front."
  default     = false
}

variable "pubsub_push_audience" {
  type        = string
  description = "OIDC audience the worker requires on Pub/Sub push requests."
  default     = "rag-worker"
}

variable "deletion_protection" {
  type        = bool
  description = "Set true once a real environment should not be destroyable by a casual apply."
  default     = false
}

variable "api_min_instances" {
  type        = number
  description = "0 keeps an idle portfolio deploy cheap. 1 removes most cold starts."
  default     = 0
}

variable "api_max_instances" {
  type        = number
  description = "Hard cap so a spike cannot outrun the Vertex quota or the bill. Tune after a load test."
  default     = 5
}

variable "worker_max_instances" {
  type    = number
  default = 3
}
