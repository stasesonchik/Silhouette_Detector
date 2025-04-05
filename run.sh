#!/bin/env sh

uvicorn silhouette_detector:detector.app --reload --port 8000 --host 0.0.0.0 && bash
