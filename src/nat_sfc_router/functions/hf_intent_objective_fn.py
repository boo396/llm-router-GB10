import json
from typing import Any, Dict, List
from transformers import AutoTokenizer
import logging
from typing import Tuple, List, Optional
import asyncio
from functools import lru_cache
import time
import requests
import os 
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Remote model configuration
REMOTE_MODEL_URL = os.getenv("ROUTER_MODEL_URL")
MODEL_NAME = os.getenv("ROUTER_MODEL_NAME")
MODEL_REVISION = os.getenv("ROUTER_MODEL_REVISION", "main")
PHI4_MULTIMODAL_URL = os.getenv("PHI4_MULTIMODAL_URL")
PHI4_REASONING_URL = os.getenv("PHI4_REASONING_URL", REMOTE_MODEL_URL)
ENABLE_MODEL_SWITCHING = os.getenv("ENABLE_MODEL_SWITCHING", "true").lower() in {"1", "true", "yes"}

# Only need tokenizer for prompt encoding (model is remote)
tokenizer = None

def _load_tokenizer():
    """Lazy load the tokenizer on first use."""
    global tokenizer
    if tokenizer is None:
        logger.info(f"Loading tokenizer for {MODEL_NAME}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
                trust_remote_code=False,
            )
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    return tokenizer

def _check_remote_model():
    """Check if remote model is available."""
    try:
        response = requests.get(f"{REMOTE_MODEL_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"Remote model at {REMOTE_MODEL_URL} is available")
            return True
    except Exception as e:
        logger.error(f"Remote model at {REMOTE_MODEL_URL} is not available: {e}")
        return False


def _is_endpoint_ready(base_url: str) -> bool:
    """Check if an OpenAI-compatible endpoint is ready."""
    if not base_url:
        return False
    for path in ("/health", "/v1/models"):
        try:
            response = requests.get(f"{base_url}{path}", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            continue
    return False


def _maybe_switch_models(target: str) -> None:
    """Start the target model container and stop the other to free GPU memory."""
    if not ENABLE_MODEL_SWITCHING:
        return

    try:
        import docker  # type: ignore
    except Exception as e:
        logger.warning(f"Model switching disabled (docker SDK unavailable): {e}")
        return

    target = target.lower()
    desired = "phi4-multimodal" if target == "multimodal" else "phi4-reasoning"
    other = "phi4-reasoning" if target == "multimodal" else "phi4-multimodal"
    desired_url = PHI4_MULTIMODAL_URL if target == "multimodal" else PHI4_REASONING_URL

    try:
        client = docker.from_env()
    except Exception as e:
        logger.warning(f"Model switching disabled (cannot connect to docker): {e}")
        return

    def _get_container(name: str):
        try:
            return client.containers.get(name)
        except Exception:
            return None

    other_container = _get_container(other)
    desired_container = _get_container(desired)

    if desired_container and desired_container.status != "running":
        logger.warning("Switching models: starting %s (this can take a few minutes). Consider adding another GB10 for concurrency.", desired)
        desired_container.start()

    if other_container and other_container.status == "running":
        logger.warning("Switching models: stopping %s to free GPU memory.", other)
        try:
            other_container.stop(timeout=180)
        except Exception as e:
            logger.warning(f"Failed to stop {other}: {e}")

    # Wait briefly for readiness
    if desired_url:
        for _ in range(60):
            if _is_endpoint_ready(desired_url):
                return
            time.sleep(2)

# Please use our provided prompt for best performance
TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""

FORMAT_PROMPT = """
Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags.  Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent. 
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"} 
"""

# Custom JSON encoder for Pydantic models and non-serializable objects
class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # Handle dict-like objects
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Handle iterables (except strings)
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return list(obj)
            except TypeError:
                pass
        return super().default(obj)

# Define route config
route_config = [
    {
        "name": "hard_question",
        "description": "A question that requires deep reasoning, or complex problem solving, or if the user asks for careful thinking or careful consideration",
    },
    {
        "name": "complex_reasoning",
        "description": "Advanced reasoning tasks such as complex logic, multi-step math, or rigorous analysis.",
    },
    {
        "name": "chit_chat",
        "description": "Any social chit chat, small talk, or casual conversation.",
    },
    {
        "name": "try_again",
        "description": "Only if the user explicitly says the previous answer was incorrect or incomplete.",
    },
    {
        "name": "image_understanding",
        "description": "A question that requires understanding an image.",
    },
    {
        "name": "image_question",
        "description": "A question that requires the assistant to see the user eg a question about their appearance, environment, scene or surroundings.",
    },
    {
        "name": "document_understanding",
        "description": "OCR, reading text from images, or understanding documents and forms.",
    },
    {
        "name": "chart_understanding",
        "description": "Understanding charts, plots, tables, or diagrams from images.",
    },
    {
        "name": "multi_image_analysis",
        "description": "Comparing multiple images or summarizing multi-image content.",
    },
    {
        "name": "audio_transcription",
        "description": "Transcribing or translating audio, or answering questions about audio content.",
    },
]

# Pre-compute routes JSON once to avoid repeated serialization
_ROUTES_JSON_CACHED = json.dumps(route_config, cls=PydanticEncoder)

MAP_INTENT_TO_PIPELINE = {
    "other": "microsoft/phi-4",
    "chit_chat": "microsoft/phi-4",
    "hard_question": "microsoft/phi-4",
    "complex_reasoning": "microsoft/phi-4",
    "image_understanding": "microsoft/Phi-4-multimodal-instruct",
    "image_question": "microsoft/Phi-4-multimodal-instruct",
    "document_understanding": "microsoft/Phi-4-multimodal-instruct",
    "chart_understanding": "microsoft/Phi-4-multimodal-instruct",
    "multi_image_analysis": "microsoft/Phi-4-multimodal-instruct",
    "audio_transcription": "microsoft/Phi-4-multimodal-instruct",
    "try_again": "microsoft/phi-4",
}

# Helper function to redact images while preserving text context
def redact_images_from_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove image data for router."""
    redacted = []
    for i, msg in enumerate(conversation):
        msg_copy = msg.copy()
        content = msg_copy.get("content")
    
        
        # If content is a list (multimodal), process it
        if isinstance(content, list):
            text_parts = []
            saw_image = False
            
            for item in content:
                logger.info(f"  Item: {type(item)}, {item if not isinstance(item, dict) else list(item.keys())}")
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        item_text = item.get("text", "")
                        text = f"<new msg>{item_text} </msg>"
                        text_parts.append(text)
                    elif item.get("type") == "image_url":
                        saw_image = True
                        continue
                        
            
            # Combine text parts and add image indicator if present
            combined_text = " ".join(text_parts).strip()

            if not combined_text and saw_image:
                combined_text = "<new msg>[image_only_message]</msg>"

            msg_copy["content"] = combined_text
        
        redacted.append(msg_copy)
    
    return redacted

# Helper function to create the system prompt for our model
def format_prompt(conversation: List[Dict[str, Any]]):
    """Create the system prompt - uses pre-computed routes JSON for efficiency."""
    return (
        TASK_INSTRUCTION.format(
            routes=_ROUTES_JSON_CACHED,  # Use pre-computed JSON
            conversation=json.dumps(conversation, cls=PydanticEncoder)
        )
        + FORMAT_PROMPT
    )

# Cached JSON response parsing
@lru_cache(maxsize=128)
def _parse_route_response(response: str) -> str:
    """Parse and cache route responses to avoid repeated JSON parsing."""
    try:
        return json.loads(response)["route"]
    except json.JSONDecodeError:
        # Handle single quote format
        import ast
        try:
            return ast.literal_eval(response)["route"]
        except Exception:
            import re
            match = re.search(r"\{\s*\"route\"\s*:\s*\"([^\"]+)\"\s*\}", response)
            if match:
                return match.group(1)
            return "other"



class HFIntentObjectiveConfig(FunctionBaseConfig, name="hf_intent_objective_fn"):
    """HF intent objective function for best route."""
    pass


def materialize_iterator(obj):
    """Recursively convert ValidatorIterator and other iterables to lists."""
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
        try:
            return [materialize_iterator(item) for item in obj]
        except TypeError:
            pass
    elif isinstance(obj, dict):
        return {k: materialize_iterator(v) for k, v in obj.items()}
    return obj

@register_function(config_type=HFIntentObjectiveConfig)
async def hf_intent_objective_fn(config: HFIntentObjectiveConfig,
                                 _builder: Builder):
    """HF intent objective function for best route."""

    from nat_sfc_router.schema.openai_chat_request import OpenAIChatRequest

    # Check if remote model is available
    _check_remote_model()
    
    # Load tokenizer (model is remote)
    loaded_tokenizer = _load_tokenizer()

    def get_route_from_conversation(conversation: List[Dict[str, Any]]) -> str:
        """Determine the best route for the conversation (using remote model)."""
        inference_start = time.perf_counter()
        logger.info("Routing with Intent Router")
        
        # Redact images from messages because the router does not support them
        # But it can still determine if the text intent requires image understanding
        redacted_conversation = redact_images_from_conversation(conversation)
        
        # ===== FORMAT PROMPT =====
        prompt_start = time.perf_counter()
        route_prompt = format_prompt(redacted_conversation)
        prompt_time = time.perf_counter() - prompt_start
        
        # ===== CONSTRUCT MESSAGES =====
        construct_start = time.perf_counter()
        messages = [
            {"role": "user", "content": route_prompt},
        ]
        construct_time = time.perf_counter() - construct_start

        # ===== ENCODE (TOKENIZE) =====
        # Not needed for remote API, but keeping for timing consistency
        encode_start = time.perf_counter()
        encode_time = time.perf_counter() - encode_start

        # ===== GENERATION (REMOTE API CALL) =====
        generation_start = time.perf_counter()
        try:
            # Call remote vLLM OpenAI-compatible API
            response = requests.post(
                f"{REMOTE_MODEL_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "max_tokens": 32,
                    "temperature": 0.3,
                    "top_p": 0.9,
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to call remote model: {e}")
            raise
        
        generation_time = time.perf_counter() - generation_start

        # ===== DECODING =====
        decode_start = time.perf_counter()
        # Response is already decoded text from remote API
        decode_time = time.perf_counter() - decode_start
        
        # Use cached parser
        route = _parse_route_response(response_text)
        
        total_time = time.perf_counter() - inference_start
        
        # Log timing breakdown
        logger.info(
            f"Route inference timing breakdown | "
            f"Format: {prompt_time*1000:.2f}ms | "
            f"Construct: {construct_time*1000:.2f}ms | "
            f"Encode: {encode_time*1000:.2f}ms | "
            f"Generate: {generation_time*1000:.2f}ms | "
            f"Decode: {decode_time*1000:.2f}ms | "
            f"Total: {total_time*1000:.2f}ms"
        )
        logger.debug(f"Route: {route}, Response: {response_text[:100]}")
        
        return route

    async def _response_fn(chat_request: OpenAIChatRequest) -> Tuple[str, str]:  # pyright: ignore[reportUnusedParameter]
        """HF intent objective function for best route."""
        response_start = time.perf_counter()

        # ===== EXTRACT MESSAGES =====
        extract_start = time.perf_counter()
        messages = chat_request.messages
        extract_time = time.perf_counter() - extract_start

        if messages:
            # ===== CONVERT TO DICT =====
            dict_convert_start = time.perf_counter()
            last_msg = messages[-1]
            last_msg_dict = last_msg.model_dump() if hasattr(last_msg, 'model_dump') else dict(last_msg)
            dict_convert_time = time.perf_counter() - dict_convert_start

            # ===== MATERIALIZE ITERATORS =====
            materialize_start = time.perf_counter()
            last_msg_dict = materialize_iterator(last_msg_dict)
            materialize_time = time.perf_counter() - materialize_start

            # Assign a list containing only the last message's dictionary
            messages_dict = [last_msg_dict]
            
            logger.debug(
                f"Message preparation timing | "
                f"Extract: {extract_time*1000:.2f}ms | "
                f"Dict convert: {dict_convert_time*1000:.2f}ms | "
                f"Materialize: {materialize_time*1000:.2f}ms"
            )
        else:
            # Handle the case where the list of messages is empty
            messages_dict = []
            logger.warning("No messages received in chat request")

        # If the latest message contains an image, force multimodal routing
        has_image = False
        if messages_dict:
            content = messages_dict[0].get("content")
            if isinstance(content, list):
                has_image = any(
                    isinstance(item, dict) and item.get("type") == "image_url"
                    for item in content
                )

        # Run model inference (blocking call in event loop)
        if has_image:
            _maybe_switch_models("multimodal")
            user_intent = "image_understanding"
        else:
            _maybe_switch_models("reasoning")
            user_intent = get_route_from_conversation(messages_dict)
        
        total_response_time = time.perf_counter() - response_start

        logger.warn(f"User intent: {user_intent} (total response time: {total_response_time*1000:.2f}ms)")
        return MAP_INTENT_TO_PIPELINE[user_intent], ""
    

    yield FunctionInfo.from_fn(
        _response_fn,
        description="Demonstrative objective function for best model.")
