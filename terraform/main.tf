data "google_project" "current" {
  project_id = var.project_id
}

locals {
  services = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
  ])
  image = (
    var.image != ""
    ? var.image
    : "${var.region}-docker.pkg.dev/${var.project_id}/${var.name_prefix}/rag-service:latest"
  )
  pubsub_agent   = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
  run_agent      = "serviceAccount:service-${data.google_project.current.number}@serverless-robot-prod.iam.gserviceaccount.com"
  topic_name     = "projects/${var.project_id}/topics/${google_pubsub_topic.documents.name}"
  needs_database = var.vector_store == "pgvector" || (var.vector_store == "vertex" && var.vertex_catalog_database)
  shared_env = {
    EMBEDDING_DIMENSIONALITY = tostring(var.embedding_dimensionality)
    EMBEDDING_MODEL          = var.embedding_model
    EMBEDDING_PROVIDER       = "vertex"
    GCP_LOCATION             = var.region
    GCP_PROJECT              = var.project_id
    GENERATION_MODEL         = var.generation_model
    GENERATION_PROVIDER      = "vertex"
    LOG_FORMAT               = "json"
    VECTOR_STORE             = var.vector_store
  }
}

resource "google_project_service" "services" {
  for_each           = local.services
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  project       = var.project_id
  location      = var.region
  repository_id = var.name_prefix
  format        = "DOCKER"
  description   = "Images for the RAG API and worker."
  depends_on    = [google_project_service.services]
}

resource "google_artifact_registry_repository_iam_member" "cloudrun_reader" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.images.repository_id
  role       = "roles/artifactregistry.reader"
  member     = local.run_agent
}

resource "google_secret_manager_secret" "database_url" {
  project   = var.project_id
  secret_id = "${var.name_prefix}-database-url"
  replication {
    auto {}
  }
  depends_on = [google_project_service.services]
}
