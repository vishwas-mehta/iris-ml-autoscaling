# IRIS ML Autoscaling Pipeline

**Week 7 MLOps Assignment | Vishwas Mehta | 22F3001150**

Kubernetes autoscaling demonstration for IRIS classification API with automated CI/CD, Docker deployment, and load testing.

## Assignment Goals

✅ CI/CD pipeline with GitHub Actions  
✅ Load testing with wrk (>1000 requests)  
✅ Kubernetes HPA (min: 1 pod, max: 3 pods)  
✅ Bottleneck demonstration at 2000 concurrent requests  

## Tech Stack

Python • Flask • scikit-learn • Docker • Kubernetes (GKE) • GitHub Actions • wrk • Google Artifact Registry

## Stress Test Results

| Test | Load | Pods | Req/sec | Latency | Result |
|------|------|------|---------|---------|---------|
| Test 1 | 1000 req | 1 | 16.35 | 1.07s | ⚠️ Baseline |
| Test 2 | 1000 req | 3 | 47.95 | 458ms | ✅ +193% throughput |
| Test 3 | 2000 req | 1 | 20.64 | 1.03s | ❌ **BOTTLENECK** |
| Test 4 | 2000 req | 3 | 53.23 | 491ms | ✅ +158% throughput |

## Key Findings

**Test 2 vs Test 1:** Autoscaling improved throughput by 193% and reduced latency by 57%

**Test 3 (Bottleneck):** Single pod overwhelmed with 200 connections - severe performance degradation with 582 timeouts

**Test 4 vs Test 3:** Autoscaling eliminated bottleneck - 158% throughput increase, 52% latency reduction

## Project Structure

iris-ml-autoscaling/
├── train.py # Model training
├── app.py # Flask API
├── Dockerfile # Container
├── k8s/
│ ├── deployment.yaml # K8s deployment
│ ├── service.yaml # LoadBalancer
│ └── hpa.yaml # Autoscaler (1-3 pods)
└── .github/workflows/
└── deploy.yml # CI/CD + stress tests


## Setup

Create GKE cluster
gcloud container clusters create iris-cluster --zone us-central1-a --num-nodes 3

Create Artifact Registry
gcloud artifacts repositories create iris-repo --location=us-central1 --repository-format=docker

Install Metrics Server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml


## GitHub Secrets Required

- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_SA_KEY`: Service account JSON key

## Conclusion

Successfully demonstrated Kubernetes autoscaling with 2-3x performance improvement. Clear bottleneck observation when restricted to 1 pod, resolved by HPA scaling to 3 pods.

---

**Date:** November 12, 2025
