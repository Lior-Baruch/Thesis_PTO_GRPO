"""core - the shared trainer layer for Exp4_OpenStack.

Submodules are imported explicitly (``from core.oracle import OracleConfig``) rather than
re-exported here, because several of them pull in torch and the read-only EDA imports the
light ones. A package-level re-export would make ``import core.timing`` drag in CUDA.
"""
