import logging
import json
import regex as re
from typing import List, Optional, Union, Dict, Any, Set


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# UUID Extraction Configuration
# -----------------------------------------------------------------

# Regex pattern to find UUIDs (standard format: 8-4-4-4-12 hex chars)
# The pattern is kept slightly permissive to catch typical UUID fields in JSON or text.
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", 
    re.IGNORECASE
)

# -----------------------------------------------------------------
# Core Utility Function
# -----------------------------------------------------------------

def _robust_extract_json(text: str) -> Optional[Any]:
    """
    Attempts to extract a complete and valid JSON object from text.
    (Duplicated from retrieval_utils.py for independence, but recommended to centralize this in a real project)
    """
    if not text:
        return None

    cleaned_text = text.strip()

    # Remove code fences (```json or ```)
    cleaned_text = re.sub(r'^\s*```(?:json)?\s*', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(r'\s*```\s*$', '', cleaned_text, flags=re.MULTILINE)

    # Remove any leading "json" or similar tokens
    cleaned_text = re.sub(r'^\s*json[:\s]*', '', cleaned_text, flags=re.IGNORECASE)

    # Find first '{' or '[' as potential JSON start
    brace_idx = min(
        [idx for idx in (cleaned_text.find('{'), cleaned_text.find('[')) if idx != -1],
        default=-1
    )
    if brace_idx == -1:
        return None

    candidate = cleaned_text[brace_idx:].strip()

    # Try JSONDecoder.raw_decode
    try:
        decoder = json.JSONDecoder()
        obj, end_idx = decoder.raw_decode(candidate)
        return obj
    except json.JSONDecodeError:
        # Fallback: try to find last closing brace and clean trailing commas
        try:
            last_brace = max(candidate.rfind('}'), candidate.rfind(']'))
            if last_brace != -1:
                json_str = candidate[:last_brace + 1]
                # remove trailing commas before closing braces/brackets
                json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
                return json.loads(json_str)
        except Exception:
            pass

    return None


def extract_uuids_from_llm_response(
    llm_response_text: str, 
    key_hint: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """
    Extracts all unique UUIDs (v4) from an LLM text response. 
    It first tries to robustly extract a JSON object, then searches for UUIDs 
    in specific fields (if key_hint is provided) or in the entire text.
    
    Args:
        llm_response_text: The raw text response from the LLM.
        key_hint: Optional key name(s) to search for UUIDs within the extracted JSON.
                  E.g., "Chunk_UUIDs", "evidence_ids".
                  
    Returns:
        A list of unique UUID strings found.
    """
    unique_uuids: Set[str] = set()
    key_hints = [key_hint] if isinstance(key_hint, str) else (key_hint or [])
    
    # 1. Try to extract JSON object
    llm_output_json = _robust_extract_json(llm_response_text)

    if llm_output_json:
        # 1a. Search within JSON for key_hint
        if key_hints:
            search_data = llm_output_json
            
            # Simple recursive search helper for keys
            def find_values_for_keys(data, keys: List[str]):
                values = []
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k in keys:
                            if isinstance(v, list):
                                values.extend(v)
                            elif isinstance(v, str):
                                values.append(v)
                        # Recursive search in sub-dictionaries/lists
                        values.extend(find_values_for_keys(v, keys))
                elif isinstance(data, list):
                    for item in data:
                        values.extend(find_values_for_keys(item, keys))
                return values
            
            all_values = find_values_for_keys(search_data, key_hints)
            
            for value in all_values:
                if isinstance(value, str):
                    # Check if the value itself is a UUID
                    if UUID_PATTERN.fullmatch(value):
                        unique_uuids.add(value)
                    else:
                        # Or search for UUIDs within the string value
                        found_in_value = UUID_PATTERN.findall(value)
                        unique_uuids.update(found_in_value)

            if unique_uuids:
                logger.info(f"✅ Found {len(unique_uuids)} UUIDs via key hint(s): {', '.join(key_hints)}")
                return list(unique_uuids)

        # 1b. If no key_hint or search failed, use the entire JSON string representation for regex search
        # This catches UUIDs that might be in an unexpected field or nested deep.
        text_to_search = json.dumps(llm_output_json)
    else:
        # 2. If no JSON extracted, search the entire raw text
        text_to_search = llm_response_text

    # 3. Final regex search on the relevant text
    found_uuids = UUID_PATTERN.findall(text_to_search)
    unique_uuids.update(found_uuids)
    
    if unique_uuids:
        logger.info(f"🔍 Found {len(unique_uuids)} UUIDs via regex match.")

    return list(unique_uuids)