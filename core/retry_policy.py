# core/retry_policy.py
import time
import random
from dataclasses import dataclass
from typing import Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetryResult:
    success: bool
    result: Optional[Any]
    reason: str
    attempts: int


class RetryPolicy:
    """
    Adaptive Retry System for SEAM PDCA assessments (เวอร์ชันแก้แล้ว 100%)
    รองรับทุกกรณีที่ LLM คืนค่าแปลก ๆ (int, str, dict ไม่ครบ key)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        jitter: bool = True,
        escalate_context: bool = True,
        shorten_prompt_on_fail: bool = True,
        exponential_backoff: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.jitter = jitter
        self.escalate_context = escalate_context
        self.shorten_prompt_on_fail = shorten_prompt_on_fail
        self.exponential_backoff = exponential_backoff

    def run(
        self,
        fn: Callable,
        *,
        level: int,
        statement: Optional[str] = None,
        context_blocks: Optional[Any] = None,
        logger: Optional[Any] = None
    ) -> RetryResult:
        current_context = context_blocks if context_blocks else {}
        current_statement = statement or ""
        last_failure_reason = "unknown"
        result: Any = None

        for attempt in range(1, self.max_attempts + 1):
            if logger:
                logger.warning(f"RetryPolicy] L{level} Attempt {attempt}/{self.max_attempts} running assessment...")

            try:
                # เรียก function (รองรับทั้งแบบมี attempt และไม่มี)
                try:
                    result = fn(attempt=attempt)
                except TypeError:
                    result = fn(
                        level=level,
                        statement=current_statement,
                        context_blocks=current_context,
                        attempt=attempt
                    )

                if result is None:
                    result = {}

                # ------------------------------------------------------------
                # ตรวจสอบผลลัพธ์แบบยืดหยุ่น (รองรับทุกกรณีจริง)
                # ------------------------------------------------------------
                if isinstance(result, dict):
                    # ถ้ามี key ที่บ่งบอกถึงผลลัพธ์ → ถือว่าผ่าน
                    if any(key in result for key in ["is_passed", "score", "level", "status"]):
                        return RetryResult(success=True, result=result, reason="normal", attempts=attempt)
                    else:
                        last_failure_reason = "LLM output missing expected keys (is_passed/score/level/status)"
                else:
                    # Fallback สำหรับ int / float / str ที่เป็นเลข
                    try:
                        if isinstance(result, (int, float)):
                            level_num = int(result)
                            result = {
                                "level": level_num,
                                "is_passed": level_num >= 3,
                                "score": level_num,
                                "status": "PASS" if level_num >= 3 else "FAIL"
                            }
                            if logger:
                                logger.info(f"RetryPolicy] LLM returned raw number {level_num} → converted to PASS")
                            return RetryResult(success=True, result=result, reason="fallback_from_int", attempts=attempt)

                        elif isinstance(result, str) and result.strip().isdigit():
                            level_num = int(result.strip())
                            result = {
                                "level": level_num,
                                "is_passed": level_num >= 3,
                                "score": level_num,
                                "status": "PASS" if level_num >= 3 else "FAIL"
                            }
                            if logger:
                                logger.info(f"RetryPolicy] LLM returned number string '{result}' → converted")
                            return RetryResult(success=True, result=result, reason="fallback_from_str", attempts=attempt)
                    except Exception as conv_e:
                        last_failure_reason = f"Conversion fallback failed: {conv_e}"

                    last_failure_reason = f"Non-dict output from LLM: {type(result).__name__} = {repr(result)}"

                # ถ้าถึงตรงนี้ → ถือว่า malformed → บังคับ retry
                if logger:
                    logger.error(f"L{level} Malformed output. Forcing retry {attempt}. Reason: {last_failure_reason}")
                raise ValueError(last_failure_reason)

            except Exception as e:
                last_failure_reason = str(e) or "unknown_error"
                if logger:
                    logger.error(f"L{level} Retry {attempt} failed: {last_failure_reason}")

                if attempt >= self.max_attempts:
                    return RetryResult(
                        success=False,
                        result=result or {"is_passed": False, "score": 0},
                        reason=last_failure_reason,
                        attempts=attempt
                    )

                # Adaptive strategies
                if self.escalate_context and logger:
                    logger.warning(f"Context escalation requested. The assessment function (fn) should use 'attempt={attempt}' to retrieve more evidence/metadata.")

                if self.shorten_prompt_on_fail and current_statement:
                    try:
                        ratios = [0.8, 0.5, 0.25]
                        idx = min(attempt - 1, len(ratios) - 1)
                        new_len = max(50, int(len(current_statement) * ratios[idx]))
                        current_statement = current_statement[:new_len]
                        if logger:
                            logger.warning(f"Shortening statement to {new_len} chars for next retry.")
                            logger.warning("Note: For true semantic preservation, pre-summarize statement before retry.")
                    except Exception:
                        pass

                # Backoff
                delay = self.base_delay * (2 ** (attempt - 1)) if self.exponential_backoff else self.base_delay * attempt
                if self.jitter:
                    delay += random.uniform(0.0, 0.8)
                if logger:
                    logger.warning(f"Waiting {delay:.2f}s before next retry...")
                time.sleep(delay)

        # Final fallback
        return RetryResult(
            success=False,
            result={"is_passed": False, "score": 0, "status": "FAIL"},
            reason=last_failure_reason,
            attempts=self.max_attempts
        )