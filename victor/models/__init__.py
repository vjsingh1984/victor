"""Shipped model artifacts (FEP-0012 universal baseline).

This package directory holds the shipped classifier artifact
(``edge_classifier_v1.npz``) so that ``pip install victor-ai`` includes it.
The ``auto`` decision backend loads it at startup via
``LocalClassifierDecisionService`` (``victor/agent/services/local_classifier_service.py``).
"""
