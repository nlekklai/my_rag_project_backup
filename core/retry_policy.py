# core/retry_policy.py
import time
import random
from dataclasses import dataclass
from typing import Any, Optional, Callable
import math
# logging ถูกนำออกไปตามโครงสร้างเดิมของผู้ใช้

@dataclass
class RetryResult:
    success: bool
    result: Optional[Any]
    reason: str
    attempts: int


class RetryPolicy:
    """
    Adaptive Retry System for multi-level SEAM PDCA assessments.
    Features:
    - Exponential backoff delay with optional jitter
    - Dynamic context escalation via 'attempt' argument
    - Statement shortening (% truncation) if previous attempt failed
    - Level-specific logging
    - Fallback for malformed outputs
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
        """
        Run the given LLM evaluation function with retry logic.

        Args:
            fn: callable that executes the LLM assessment. (Must accept 'attempt: int')
            level: current PDCA level (1–5)
            statement: statement data to evaluate (used for shortening)
            context_blocks: context to provide to the LLM
            logger: logging instance (must implement .warning and .error)

        Returns:
            RetryResult
        """

        current_context = context_blocks if context_blocks else ""
        current_statement = statement if statement else ""
        last_failure_reason = "unknown"
        result: Optional[dict] = None # Initialization for safety

        for attempt in range(1, self.max_attempts + 1):
            if logger:
                # Level-specific logging
                logger.warning(f"🔄 [RetryPolicy] L{level} Attempt {attempt}/{self.max_attempts} running assessment...")

            try:
                # Execute function safely
                if callable(fn):
                    try:
                        # Attempt to call with 'attempt' argument
                        result = fn(attempt=attempt)
                    except TypeError:
                        # Fallback with known args if fn doesn't match the signature
                        result = fn(
                            level=level,
                            statement=current_statement,
                            context_blocks=current_context,
                            attempt=attempt
                        )
                else:
                    raise ValueError("Provided fn is not callable")

                if result is None:
                    result = {}

                # Validate structured result
                if isinstance(result, dict) and ("status" in result and result["status"] in ["PASS", "FAIL"]):
                    return RetryResult(success=True, result=result, reason="normal", attempts=attempt)

                # Malformed output fallback → force retry
                last_failure_reason = "Malformed LLM output - Status key missing or invalid. (Forcing Retry Test)"

                if logger:
                    logger.error(f"❌ L{level} Malformed output detected. Forcing Retry {attempt}.")

                # เปลี่ยนจาก return → raise เพื่อเข้า Adaptive retry logic
                raise ValueError(last_failure_reason)


            except Exception as e:
                last_failure_reason = str(e)
                if logger:
                    logger.error(f"❌ L{level} Retry {attempt} failed: {last_failure_reason}")

                if attempt >= self.max_attempts:
                    return RetryResult(
                        success=False,
                        result=result,
                        reason=last_failure_reason,
                        attempts=attempt
                    )

                # -------- Adaptive retry strategies --------

                # 1. Context escalation hint
                if self.escalate_context and logger:
                    logger.warning(f"📈 Context escalation requested. The assessment function (fn) should use 'attempt={attempt}' to retrieve more evidence/metadata.")

                # 2. Dynamic statement shortening (% truncation)
                if self.shorten_prompt_on_fail and current_statement:
                    try:
                        # Truncation ratios: 80% on retry 1, 50% on retry 2, 25% on retry 3
                        truncate_ratio = [0.8, 0.5, 0.25]
                        idx = min(attempt - 1, len(truncate_ratio) - 1)
                        # Ensure the statement is not shorter than 50 characters (arbitrary min)
                        new_length = max(50, int(len(current_statement) * truncate_ratio[idx]))
                        current_statement = current_statement[:new_length]
                        if logger:
                            logger.warning(f"✂️ Shortening statement to {new_length} chars for next retry.")
                            logger.warning("💡 Note: For true semantic preservation, pre-summarize statement before retry.")
                    except Exception:
                        current_statement = str(current_statement)

                # 3. Exponential backoff + optional jitter
                if self.exponential_backoff:
                    # Exponential: base_delay * 2^(attempt-1)
                    delay = self.base_delay * (2 ** (attempt - 1))
                    log_type = "Exponential"
                else:
                    # Linear
                    delay = self.base_delay * attempt
                    log_type = "Linear"

                if self.jitter:
                    # Add uniform jitter between 0.0 and 0.8 seconds
                    jitter_val = random.uniform(0.0, 0.8)
                    delay += jitter_val

                if logger:
                    logger.warning(f"⏳ Waiting ({log_type}) {delay:.2f}s before next retry...")
                time.sleep(delay)

        # Final fallback return
        return RetryResult(success=False, result=None, reason=last_failure_reason, attempts=self.max_attempts)