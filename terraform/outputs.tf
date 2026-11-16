output "api_uri" {
  description = "Cloud Run URL of the API. IAM still has to allow the caller."
  value       = google_cloud_run_v2_service.api.uri
}

output "worker_uri" {
  description = "Cloud Run URL of the worker. Only the push service account can invoke it."
  value       = google_cloud_run_v2_service.worker.uri
}

output "image_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/rag-service"
}

output "documents_topic" {
  value = local.topic_name
}

output "api_service_account" {
  value = google_service_account.api.email
}

output "worker_service_account" {
  value = google_service_account.worker.email
}

output "database_secret" {
  value = google_secret_manager_secret.database_url.secret_id
}
