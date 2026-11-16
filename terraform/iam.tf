resource "google_service_account" "api" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-api"
  display_name = "RAG API runtime"
}

resource "google_service_account" "worker" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-worker"
  display_name = "RAG ingest worker"
}

resource "google_service_account" "pubsub_push" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-push"
  display_name = "Pub/Sub push identity for the RAG worker"
}

# roles/aiplatform.user is the narrowest predefined role that covers Gemini,
# embeddings, and Vector Search query/upsert together. A custom role limited
# to the methods this service actually calls is a follow-up once a deployed
# project shows the exact permission set.
resource "google_project_iam_member" "api_aiplatform" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "worker_aiplatform" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

locals {
  observability_members = {
    api_logs       = { member = google_service_account.api.email, role = "roles/logging.logWriter" }
    api_metrics    = { member = google_service_account.api.email, role = "roles/monitoring.metricWriter" }
    api_trace      = { member = google_service_account.api.email, role = "roles/cloudtrace.agent" }
    worker_logs    = { member = google_service_account.worker.email, role = "roles/logging.logWriter" }
    worker_metrics = { member = google_service_account.worker.email, role = "roles/monitoring.metricWriter" }
    worker_trace   = { member = google_service_account.worker.email, role = "roles/cloudtrace.agent" }
  }
}

resource "google_project_iam_member" "observability" {
  for_each = local.observability_members
  project  = var.project_id
  role     = each.value.role
  member   = "serviceAccount:${each.value.member}"
}

resource "google_secret_manager_secret_iam_member" "api_database" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.database_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

resource "google_secret_manager_secret_iam_member" "worker_database" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.database_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.worker.email}"
}
