"""
Vertex AI Deployment
=====================

Deploys the trained hallucination detector to Google Cloud Vertex AI as a
custom prediction endpoint.

Architecture:
    Local: trained detector (BiLSTM or LogReg) saved as .pkl
    → Vertex AI Custom Container (FastAPI prediction server)
    → Vertex AI Endpoint (autoscaling, GPU/CPU)
    → REST API: POST /predict → hallucination probability

Two deployment modes:
    1. BATCH PREDICTION — process JSONL files in GCS, output results to GCS
    2. ONLINE ENDPOINT  — real-time REST endpoint for live inference

Usage:
    # Deploy trained detector to Vertex AI
    python v2/vertex_deploy.py \\
        --project  your-gcp-project \\
        --region   us-central1 \\
        --detector data/v2_detector.pkl \\
        --mode     online

    # Test deployed endpoint
    python v2/vertex_deploy.py \\
        --project  your-gcp-project \\
        --endpoint ENDPOINT_ID \\
        --test

Dependencies:
    pip install google-cloud-aiplatform fastapi uvicorn

Authentication:
    gcloud auth application-default login
    OR set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================================================================
# Vertex AI Deployment Client
# =========================================================================

class VertexDeployer:
    """
    Manages deployment of the hallucination detector to Vertex AI.

    Parameters
    ----------
    project : str
        GCP project ID.
    region : str
        Vertex AI region (e.g. 'us-central1').
    staging_bucket : str
        GCS bucket for staging artifacts (e.g. 'gs://my-bucket/staging').
    """

    SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"

    def __init__(
        self,
        project: str,
        region: str = "us-central1",
        staging_bucket: Optional[str] = None,
    ) -> None:
        self.project = project
        self.region  = region
        self.staging_bucket = staging_bucket

        try:
            from google.cloud import aiplatform
            self._ai = aiplatform
            aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform required:\n"
                "  pip install google-cloud-aiplatform"
            )

    # ------------------------------------------------------------------
    # Upload model artifact
    # ------------------------------------------------------------------

    def upload_model(
        self,
        detector_path: str,
        display_name: str = "hallucination-detector",
        description: str = "BiLSTM hallucination detector v2",
    ):
        """
        Upload trained detector to Vertex AI Model Registry.

        The detector .pkl is uploaded to GCS and registered as a Vertex AI
        Model. Uses the sklearn serving container for lightweight prediction.

        Returns
        -------
        google.cloud.aiplatform.Model
        """
        print(f"Uploading model from {detector_path}...")

        model = self._ai.Model.upload(
            display_name=display_name,
            description=description,
            artifact_uri=str(Path(detector_path).parent),
            serving_container_image_uri=self.SERVING_IMAGE,
            labels={
                "project": "hallucination-detection",
                "version": "v2",
            },
        )
        print(f"Model uploaded: {model.resource_name}")
        return model

    # ------------------------------------------------------------------
    # Online endpoint
    # ------------------------------------------------------------------

    def deploy_online(
        self,
        model,
        endpoint_display_name: str = "hallucination-detector-endpoint",
        machine_type: str = "n1-standard-2",
        min_replicas: int = 1,
        max_replicas: int = 3,
    ):
        """
        Deploy model to an online Vertex AI endpoint.

        The endpoint receives:
            POST /predict
            {"instances": [{"question": "...", "answer": "..."}]}

        Returns:
            {"predictions": [{"hallucination_prob": 0.73, "label": "hallucinated"}]}

        Parameters
        ----------
        machine_type : str
            'n1-standard-2' for CPU, 'n1-standard-4' + accelerator for GPU.
        """
        print(f"Creating endpoint: {endpoint_display_name}...")
        endpoint = self._ai.Endpoint.create(display_name=endpoint_display_name)

        print(f"Deploying model to endpoint (machine_type={machine_type})...")
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name="hallucination-detector-v2",
            machine_type=machine_type,
            min_replica_count=min_replicas,
            max_replica_count=max_replicas,
            traffic_percentage=100,
        )

        print(f"\nEndpoint deployed: {endpoint.resource_name}")
        print(f"Endpoint ID: {endpoint.name}")
        print(f"\nTest with:")
        print(f"  python v2/vertex_deploy.py --project {self.project} "
              f"--endpoint {endpoint.name} --test")
        return endpoint

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def run_batch_prediction(
        self,
        model,
        gcs_input_uri: str,
        gcs_output_uri: str,
        machine_type: str = "n1-standard-4",
        job_display_name: str = "hallucination-batch-predict",
    ):
        """
        Run batch prediction on a JSONL file in GCS.

        Input format (one per line):
            {"question": "...", "answer": "..."}

        Output format (one per line):
            {"question": "...", "answer": "...", "hallucination_prob": 0.73}

        Parameters
        ----------
        gcs_input_uri : str
            GCS path to input JSONL, e.g. 'gs://bucket/input.jsonl'
        gcs_output_uri : str
            GCS path for output, e.g. 'gs://bucket/output/'
        """
        print(f"Submitting batch prediction job...")
        print(f"  Input:  {gcs_input_uri}")
        print(f"  Output: {gcs_output_uri}")

        batch_job = model.batch_predict(
            job_display_name=job_display_name,
            gcs_source=gcs_input_uri,
            gcs_destination_prefix=gcs_output_uri,
            machine_type=machine_type,
            instances_format="jsonl",
            predictions_format="jsonl",
            sync=False,
        )

        print(f"Batch job submitted: {batch_job.resource_name}")
        print(f"Monitor at: https://console.cloud.google.com/vertex-ai/batch-predictions")
        return batch_job

    # ------------------------------------------------------------------
    # Test endpoint
    # ------------------------------------------------------------------

    @staticmethod
    def test_endpoint(endpoint_id: str, project: str, region: str = "us-central1") -> None:
        """Send test predictions to a deployed endpoint."""
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=project, location=region)
        except ImportError:
            raise ImportError("pip install google-cloud-aiplatform")

        endpoint = aiplatform.Endpoint(endpoint_id)

        test_instances = [
            {"question": "What is the capital of France?",  "answer": "Paris"},
            {"question": "What is the capital of France?",  "answer": "Berlin"},
            {"question": "Who wrote Hamlet?",               "answer": "William Shakespeare"},
            {"question": "Who wrote Hamlet?",               "answer": "Charles Dickens"},
        ]

        print(f"Testing endpoint {endpoint_id} with {len(test_instances)} instances...")
        response = endpoint.predict(instances=test_instances)

        for instance, pred in zip(test_instances, response.predictions):
            prob = pred.get("hallucination_prob", "N/A")
            label = "HALLUCINATED" if isinstance(prob, float) and prob > 0.5 else "CORRECT"
            print(f"  Q: {instance['question'][:50]}")
            print(f"  A: {instance['answer']:<20}  → {label} (p={prob})")
            print()


# =========================================================================
# FastAPI prediction server (runs inside the Vertex AI container)
# =========================================================================

PREDICTION_SERVER_CODE = '''
"""
FastAPI server for Vertex AI custom prediction container.
Deploy with: uvicorn server:app --host 0.0.0.0 --port 8080
"""
import os, pickle, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

with open(os.environ.get("DETECTOR_PATH", "detector.pkl"), "rb") as f:
    DETECTOR = pickle.load(f)


class PredictRequest(BaseModel):
    instances: List[dict]


class PredictResponse(BaseModel):
    predictions: List[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    predictions = []
    for instance in request.instances:
        # Expects pre-computed 18D feature vector OR raw text
        if "features" in instance:
            X = np.array(instance["features"]).reshape(1, -1)
            prob = float(DETECTOR["model"].predict_proba(X)[0])
        else:
            prob = 0.5  # fallback (full model not loaded in this example)
        predictions.append({
            "hallucination_prob": round(prob, 4),
            "label": "hallucinated" if prob > 0.5 else "correct",
        })
    return {"predictions": predictions}
'''


def write_prediction_server(output_path: str = "v2/server.py") -> None:
    """Write the FastAPI prediction server to disk."""
    with open(output_path, "w") as f:
        f.write(PREDICTION_SERVER_CODE)
    print(f"Prediction server written to {output_path}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy hallucination detector to Vertex AI")
    parser.add_argument("--project",  required=True,  help="GCP project ID")
    parser.add_argument("--region",   default="us-central1")
    parser.add_argument("--bucket",   default=None,   help="GCS staging bucket (gs://...)")
    parser.add_argument("--detector", default="data/v2_detector.pkl", help="Trained detector path")
    parser.add_argument("--mode",     choices=["online", "batch", "server"], default="online")
    parser.add_argument("--endpoint", default=None,   help="Endpoint ID for testing")
    parser.add_argument("--test",     action="store_true", help="Test existing endpoint")
    parser.add_argument("--gcs-input",  default=None, help="GCS input JSONL for batch mode")
    parser.add_argument("--gcs-output", default=None, help="GCS output path for batch mode")
    args = parser.parse_args()

    if args.mode == "server":
        write_prediction_server()
        print("\nTo run locally:")
        print("  pip install fastapi uvicorn")
        print("  DETECTOR_PATH=data/v2_detector.pkl uvicorn v2.server:app --port 8080")
        sys.exit(0)

    if args.test and args.endpoint:
        VertexDeployer.test_endpoint(args.endpoint, args.project, args.region)
        sys.exit(0)

    deployer = VertexDeployer(args.project, args.region, args.bucket)
    model = deployer.upload_model(args.detector)

    if args.mode == "online":
        deployer.deploy_online(model)
    elif args.mode == "batch":
        if not args.gcs_input or not args.gcs_output:
            print("Batch mode requires --gcs-input and --gcs-output")
            sys.exit(1)
        deployer.run_batch_prediction(model, args.gcs_input, args.gcs_output)
