# -*- coding: utf-8 -*-

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
from core.engine.iso_mapping import ISO_30401_MAPPING
from config.global_vars import (
    DEFAULT_TENANT, DEFAULT_YEAR, 
    ANALYSIS_FINAL_K, RETRIEVAL_TOP_K
)
from core.llm_data_utils import (
    retrieve_context_with_filter,
    create_context_summary_llm
)

logger = logging.getLogger(__name__)

class ISOEngine:
    def __init__(self, tenant: str = None, year: int = None, vs_manager: Any = None, llm: Any = None):
        self.tenant = tenant or DEFAULT_TENANT
        self.year = year or DEFAULT_YEAR
        self.vs_manager = vs_manager
        self.llm = llm
        self.mapping = ISO_30401_MAPPING

    async def generate_iso_report(self, seam_final_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        วิเคราะห์ ISO Readiness โดยใช้ระบบ Retrieval + Batch Reranking (จาก llm_data_utils)
        """
        start_time = time.time()
        iso_report = {
            "overall_readiness_score": 0,
            "tenant": self.tenant,
            "year": self.year,
            "clauses_analysis": {},
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # ดึงคะแนน SE-AM รายข้อเก็บไว้ใน dict เพื่อความเร็ว
        seam_scores = {
            item["sub_criteria_id"]: item 
            for item in seam_final_results.get("results", [])
        }

        total_weighted_score = 0
        
        for clause, config in self.mapping.items():
            primary = config["primary_source"]
            support = config["support_source"]

            # 1. ค้นหาหลักฐานข้าม Enabler โดยใช้ Retrieval ตัวเก่งของพี่
            # เราจะไปดึงข้อมูลจากถัง 'im' หรือ 'hcm' โดยใช้ Keywords ที่กำหนดใน Mapping
            cross_enabler_context = await self._get_robust_evidence(
                target_enabler=support["enabler"],
                keywords=config["key_evidence"]
            )

            # 2. ดึงคะแนน SE-AM ปัจจุบัน (KM)
            km_data = seam_scores.get(primary["sub"], {})
            km_score = km_data.get("score", 0.0)
            
            # 3. ให้ AI สรุปความเชื่อมโยง (Gap Analysis)
            # ใช้ create_context_summary_llm จาก utils เพื่อรักษามาตรฐานการเขียนรายงาน
            ai_analysis = "ไม่พบหลักฐานสนับสนุนจากระบบอื่น"
            if cross_enabler_context["top_evidences"]:
                ai_analysis = create_context_summary_llm(
                    context=cross_enabler_context["top_evidences"],
                    sub_criteria_name=f"ISO 30401: {clause}",
                    target_level="Compliance Analysis",
                    sub_id=primary["sub"],
                    llm=self.llm
                )

            # 4. คำนวณความพร้อม (Logic: SEAM Score + Evidence Presence)
            has_evidence = len(cross_enabler_context["top_evidences"]) > 0
            readiness = self._calculate_readiness(km_score, has_evidence)
            
            iso_report["clauses_analysis"][clause] = {
                "readiness_pct": readiness["pct"],
                "status": readiness["status"],
                "seam_reference": {
                    "sub_id": primary["sub"],
                    "score": km_score
                },
                "support_evidence": {
                    "enabler": support["enabler"],
                    "files": [e["source"] for e in cross_enabler_context["top_evidences"][:3]],
                    "found_count": len(cross_enabler_context["top_evidences"])
                },
                "ai_analysis": ai_analysis,
                "iso_requirement": config["iso_requirement"]
            }
            
            total_weighted_score += readiness["pct"]

        # สรุปคะแนนภาพรวม
        if self.mapping:
            iso_report["overall_readiness_score"] = round(total_weighted_score / len(self.mapping), 2)
        
        iso_report["execution_time"] = round(time.time() - start_time, 2)
        return iso_report

    async def _get_robust_evidence(self, target_enabler: str, keywords: List[str]) -> Dict[str, Any]:
        """
        ห่อหุ้ม retrieve_context_with_filter เพื่อทำ Cross-Enabler Retrieval
        """
        query = f"Evidence of {', '.join(keywords)} for ISO 30401 standards"
        
        # รัน Retrieval ที่มี Batch Reranking ในตัว
        # ระบุ doc_type='evidence' และสลับ enabler ไปยังถังเป้าหมาย (im/hcm)
        return retrieve_context_with_filter(
            query=query,
            doc_type="evidence",
            tenant=self.tenant,
            year=self.year,
            enabler=target_enabler,
            vectorstore_manager=self.vs_manager,
            top_k=RETRIEVAL_TOP_K # ใช้ค่าจาก global_vars ที่พี่ตั้งไว้ (เช่น 500)
        )

    def _calculate_readiness(self, score: float, has_evidence: bool) -> Dict:
        """ประเมินระดับความพร้อม"""
        # คะแนนฐานจาก SE-AM (80%) + คะแนนจากการพบหลักฐานสนับสนุน (20%)
        pct = (score / 5.0) * 80
        if has_evidence:
            pct += 20
        
        status = "Critical Gap"
        if pct >= 80: status = "Compliant / Ready"
        elif pct >= 50: status = "Partially Compliant"
        
        return {"pct": min(pct, 100), "status": status}
    

    # ตัวอย่างการเรียกใช้ใน SEAM Assessment
# วิธีการเรียกใช้ใน seam_assessment.py: แทรกโค้ดนี้ลงในส่วนท้ายของกระบวนการประเมิน
# if self.enabler == "KM":
#     from core.engine.iso_engine import ISOEngine
    
#     # สร้าง Engine
#     iso_auditor = ISOEngine(
#         tenant=self.tenant, 
#         year=self.year, 
#         vs_manager=self.vs_manager, # ใช้ตัวจัดการ Vector ของพี่
#         llm=self.llm
#     )
    
#     # รันการวิเคราะห์ (ส่งผลลัพธ์ SE-AM KM เข้าไป)
#     iso_results = await iso_auditor.generate_iso_report(final_results)
    
#     # ฝังผล ISO เข้าไปในก้อน JSON หลัก
#     final_results["iso_30401_readiness"] = iso_results