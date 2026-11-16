resource "google_cloud_run_v2_service" "api" {
  project             = var.project_id
  name                = "${var.name_prefix}-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.deletion_protection

  template {
    service_account                  = google_service_account.api.email
    timeout                          = "30s"
    max_instance_request_concurrency = 32

    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = var.api_max_instances
    }

    containers {
      image = local.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      dynamic "env" {
        for_each = local.shared_env
        content {
          name  = env.key
          value = env.value
        }
      }

      env {
        name  = "ROLE"
        value = "api"
      }

      env {
        name  = "INGEST_TRANSPORT"
        value = "pubsub"
      }

      env {
        name  = "PUBSUB_TOPIC"
        value = local.topic_name
      }

      env {
        name  = "OTEL_SERVICE_NAME"
        value = "rag-api"
      }

      dynamic "env" {
        for_each = var.vector_store == "memory" ? [1] : []
        content {
          name  = "ALLOW_MEMORY_STORE"
          value = "true"
        }
      }

      dynamic "env" {
        for_each = local.needs_database ? [1] : []
        content {
          name = "DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.database_url.secret_id
              version = "latest"
            }
          }
        }
      }

      dynamic "env" {
        for_each = var.vector_store == "vertex" ? [1] : []
        content {
          name  = "VECTOR_SEARCH_INDEX"
          value = var.vector_search_index
        }
      }

      dynamic "env" {
        for_each = var.vector_store == "vertex" ? [1] : []
        content {
          name  = "VECTOR_SEARCH_ENDPOINT"
          value = var.vector_search_endpoint
        }
      }

      dynamic "env" {
        for_each = var.vector_store == "vertex" ? [1] : []
        content {
          name  = "VECTOR_SEARCH_DEPLOYED_INDEX_ID"
          value = var.vector_search_deployed_index_id
        }
      }

      startup_probe {
        timeout_seconds   = 3
        period_seconds    = 5
        failure_threshold = 12
        http_get {
          path = "/readyz"
          port = 8080
        }
      }

      liveness_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
      }
    }
  }

  depends_on = [google_project_service.services]
}

resource "google_cloud_run_v2_service" "worker" {
  project             = var.project_id
  name                = "${var.name_prefix}-worker"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.deletion_protection

  template {
    service_account                  = google_service_account.worker.email
    timeout                          = "120s"
    max_instance_request_concurrency = 4

    scaling {
      min_instance_count = 0
      max_instance_count = var.worker_max_instances
    }

    containers {
      image = local.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle = true
      }

      dynamic "env" {
        for_each = local.shared_env
        content {
          name  = env.key
          value = env.value
        }
      }

      env {
        name  = "ROLE"
        value = "worker"
      }

      env {
        name  = "INGEST_TRANSPORT"
        value = "inline"
      }

      env {
        name  = "PUBSUB_PUSH_AUDIENCE"
        value = var.pubsub_push_audience
      }

      env {
        name  = "PUBSUB_PUSH_SERVICE_ACCOUNT"
        value = google_service_account.pubsub_push.email
      }

      env {
        name  = "OTEL_SERVICE_NAME"
        value = "rag-worker"
      }

      dynamic "env" {
        for_each = var.vector_store == "memory" ? [1] : []
        content {
          name  = "ALLOW_MEMORY_STORE"
          value = "true"
        }
      }

      dynamic "env" {
        for_each = local.needs_database ? [1] : []
        content {
          name = "DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.database_url.secret_id
              version = "latest"
            }
          }
        }
      }

      dynamic "env" {
        for_each = var.vector_store == "vertex" ? [1] : []
        content {
          name  = "VECTOR_SEARCH_INDEX"
          value = var.vector_search_index
        }
      }

      dynamic "env" {
        for_each = var.vector_store == "vertex" ? [1] : []
        content {
          name  = "VECTOR_SEARCH_ENDPOINT"
          value = var.vector_search_endpoint
        }
      }

      dynamic "env" {
        for_each = var.vector_store == "vertex" ? [1] : []
        content {
          name  = "VECTOR_SEARCH_DEPLOYED_INDEX_ID"
          value = var.vector_search_deployed_index_id
        }
      }

      startup_probe {
        timeout_seconds   = 3
        period_seconds    = 5
        failure_threshold = 12
        http_get {
          path = "/readyz"
          port = 8080
        }
      }

      liveness_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
      }
    }
  }

  depends_on = [google_project_service.services]
}

resource "google_cloud_run_v2_service_iam_member" "push_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.worker.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.pubsub_push.email}"
}

resource "google_cloud_run_v2_service_iam_member" "api_public" {
  count    = var.api_allow_unauthenticated ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
