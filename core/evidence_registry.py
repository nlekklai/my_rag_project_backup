# core/evidence_registry.py
# Production Final – Evidence Registry (Tenant-scoped)
# Author: AI Architecture Assistant
# Date: Dec 2025

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from config.global_vars import (
    DATA_STORE_ROOT,
    DEFAULT_TENANT,
    DEFAULT_YEAR,
    DEFAULT_ENABLER
)

logger = logging.getLogger(__name__)

# ============================================================
# Evidence Registry File Path
# ============================================================

def _registry_path(tenant: str) -> str:
    return os.path.join(DATA_STORE_ROOT, tenant.lower(), "evidence_registry.json")

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_uuid() -> str:
    return str(uuid.uuid4())

# ============================================================
# Core Class for Ingest / Assessment
# ============================================================

class EvidenceRegistry:
    """
    Tenant-scoped Evidence Registry handler
    """

    def __init__(self, tenant: str = DEFAULT_TENANT):
        self.tenant = tenant
        self.path = _registry_path(tenant)
        self._registry = self._load()

    def _load(self) -> Dict[str, Dict]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load evidence registry: {e}")
            return {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, ensure_ascii=False, indent=2)

    def create(self, *, year: int = DEFAULT_YEAR, enabler: str = DEFAULT_ENABLER,
               sub_criteria: str, level: int, doc_uuid: str, chunk_uuid: str,
               strength: float = 1.0, created_by: Optional[str] = None) -> str:
        evidence_id = _new_uuid()
        self._registry[evidence_id] = {
            "evidence_id": evidence_id,
            "tenant": self.tenant,
            "year": year,
            "enabler": enabler,
            "sub_criteria": sub_criteria,
            "level": level,
            "doc_uuid": doc_uuid,
            "chunk_uuid": chunk_uuid,
            "strength": strength,
            "status": "draft",
            "created_by": created_by,
            "created_at": _now(),
            "updated_at": _now()
        }
        self.save()
        logger.info(f"Evidence created: {evidence_id}")
        return evidence_id

    def update_status(self, evidence_id: str, status: str) -> bool:
        if evidence_id not in self._registry:
            return False
        self._registry[evidence_id]["status"] = status
        self._registry[evidence_id]["updated_at"] = _now()
        self.save()
        return True

    def delete(self, evidence_id: str) -> bool:
        if evidence_id not in self._registry:
            return False
        del self._registry[evidence_id]
        self.save()
        return True

    def get_for_assessment(self, *, year: int, enabler: str, sub_criteria: Optional[str] = None,
                           min_level: Optional[int] = None, status: str = "approved") -> List[Dict[str, Any]]:
        results = []
        for ev in self._registry.values():
            if ev["tenant"] != self.tenant:
                continue
            if ev["year"] != year:
                continue
            if ev["enabler"] != enabler:
                continue
            if status and ev["status"] != status:
                continue
            if sub_criteria and ev["sub_criteria"] != sub_criteria:
                continue
            if min_level and ev["level"] < min_level:
                continue
            results.append(ev)
        return results

    def get_by_document(self, doc_uuid: str) -> List[Dict[str, Any]]:
        return [ev for ev in self._registry.values() if ev["doc_uuid"] == doc_uuid]

    def get_by_chunk(self, chunk_uuid: str) -> List[Dict[str, Any]]:
        return [ev for ev in self._registry.values() if ev["chunk_uuid"] == chunk_uuid]

    def exists(self, *, year: int, enabler: str, sub_criteria: str, chunk_uuid: str) -> bool:
        for ev in self._registry.values():
            if (ev["tenant"] == self.tenant and ev["year"] == year and
                ev["enabler"] == enabler and ev["sub_criteria"] == sub_criteria and
                ev["chunk_uuid"] == chunk_uuid):
                return True
        return False
