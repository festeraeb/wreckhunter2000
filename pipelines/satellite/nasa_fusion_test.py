"""Run a quick fusion confidence test for known locations.

This script creates placeholder SAR cluster objects at known locations (Big Tub
Harbor, Little Tub Harbor, Cedarville) and runs the NASA fusion scorer.

It prints a GeoJSON FeatureCollection with a `fusion_score` for each point.

Usage:
  python scripts/nasa_fusion_test.py
"""

import os
import sys
import requests

# Ensure the repo root is on PYTHONPATH so 'sentinel_hunt' can be imported.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sentinel_hunt.python.nasa_fusion import FusionScorer, SarCluster


def fetch_swot_data(bbox):
    # Placeholder for SWOT API integration
    return {"swot_score": 0.8}

def fetch_ecostress_data(bbox):
    # Placeholder for ECOSTRESS API integration
    return {"ecostress_score": 0.7}

def fetch_opera_data(bbox):
    # Placeholder for OPERA API integration
    return {"opera_score": 0.9}

class FusionScorer:
    def score_clusters(self, clusters, start, end):
        scores = []
        for cluster in clusters:
            bbox = [
                cluster.lon - 0.1, cluster.lat - 0.1,
                cluster.lon + 0.1, cluster.lat + 0.1
            ]
            swot_data = fetch_swot_data(bbox)
            ecostress_data = fetch_ecostress_data(bbox)
            opera_data = fetch_opera_data(bbox)
            fusion_score = (swot_data["swot_score"] + ecostress_data["ecostress_score"] + opera_data["opera_score"]) / 3
            scores.append(fusion_score)
        return scores

    def score_to_geojson(self, clusters, scores):
        features = []
        for cluster, score in zip(clusters, scores):
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [cluster.lon, cluster.lat]
                },
                "properties": {
                    "cluster_id": cluster.cluster_id,
                    "persistence": cluster.persistence,
                    "confidence": cluster.confidence,
                    "fusion_score": score
                }
            })
        return {"type": "FeatureCollection", "features": features}

def main():
    clusters = [
        SarCluster(lat=45.255, lon=-81.621, persistence=0.9, confidence=0.85, cluster_id=1, props={"name": "Big Tub Harbor"}),
        SarCluster(lat=45.271, lon=-81.615, persistence=0.8, confidence=0.82, cluster_id=2, props={"name": "Little Tub Harbor"}),
        SarCluster(lat=45.6276, lon=-84.3303, persistence=0.95, confidence=0.92, cluster_id=3, props={"name": "Cedarville"}),
        SarCluster(lat=42.1520854637921, lon=-81.21619205186077, persistence=0.85, confidence=0.93, cluster_id=4, props={"name": "Lake Erie Hit 1"}),
        SarCluster(lat=42.1520854637921, lon=-80.71229746913508, persistence=0.86, confidence=0.86, cluster_id=5, props={"name": "Lake Erie Hit 2"}),
        SarCluster(lat=41.98418600239213, lon=-81.21619205186077, persistence=0.87, confidence=0.93, cluster_id=6, props={"name": "Lake Erie Hit 3"}),
    ]

    scorer = FusionScorer()
    scores = scorer.score_clusters(clusters, start="2024-09-01", end="2024-09-20")
    geojson = scorer.score_to_geojson(clusters, scores)

    print("Fusion GeoJSON:\n")
    print(geojson)


if __name__ == "__main__":
    main()
