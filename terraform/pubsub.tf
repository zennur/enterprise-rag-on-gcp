resource "google_pubsub_topic" "documents" {
  project    = var.project_id
  name       = "${var.name_prefix}-documents"
  depends_on = [google_project_service.services]
}

resource "google_pubsub_topic" "dead_letter" {
  project                    = var.project_id
  name                       = "${var.name_prefix}-documents-dead-letter"
  message_retention_duration = "604800s"
  depends_on                 = [google_project_service.services]
}

resource "google_pubsub_subscription" "worker" {
  project = var.project_id
  name    = "${var.name_prefix}-documents-worker"
  topic   = google_pubsub_topic.documents.id

  ack_deadline_seconds    = 120
  enable_message_ordering = true

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter.id
    max_delivery_attempts = 5
  }

  push_config {
    # Pub/Sub push comes from Google's public edge, so the worker is not
    # internal-only. The push service account is the only run.invoker.
    push_endpoint = "${google_cloud_run_v2_service.worker.uri}/internal/pubsub"
    oidc_token {
      service_account_email = google_service_account.pubsub_push.email
      audience              = var.pubsub_push_audience
    }
  }

  depends_on = [google_project_service.services]
}

resource "google_pubsub_subscription" "dead_letter" {
  project = var.project_id
  name    = "${var.name_prefix}-documents-dead-letter"
  topic   = google_pubsub_topic.dead_letter.id

  expiration_policy {
    ttl = ""
  }
}

resource "google_pubsub_topic_iam_member" "api_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.documents.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_pubsub_subscription_iam_member" "worker_subscriber" {
  project      = var.project_id
  subscription = google_pubsub_subscription.worker.name
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_pubsub_subscription_iam_member" "worker_dead_letter" {
  project      = var.project_id
  subscription = google_pubsub_subscription.dead_letter.name
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_pubsub_subscription_iam_member" "agent_on_source" {
  project      = var.project_id
  subscription = google_pubsub_subscription.worker.name
  role         = "roles/pubsub.subscriber"
  member       = local.pubsub_agent
}

resource "google_pubsub_topic_iam_member" "agent_on_dead_letter" {
  project = var.project_id
  topic   = google_pubsub_topic.dead_letter.name
  role    = "roles/pubsub.publisher"
  member  = local.pubsub_agent
}
