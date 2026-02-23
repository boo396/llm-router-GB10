"""
Multimodal LLM Router Demo Application

This Gradio-based web app demonstrates routing chat requests to different LLM endpoints
based on a classification made by a router endpoint.
"""

import os
import base64
import time
import gradio as gr
import requests
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configuration
ROUTER_ENDPOINT = os.getenv("ROUTER_ENDPOINT", "http://localhost:8001/sfc_router/chat/completions")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
PHI4_MULTIMODAL_ENDPOINT = os.getenv("PHI4_MULTIMODAL_ENDPOINT", "http://localhost:8012/v1")
PHI4_REASONING_ENDPOINT = os.getenv("PHI4_REASONING_ENDPOINT", "http://localhost:8013/v1")
LOCAL_OPENAI_API_KEY = os.getenv("LOCAL_OPENAI_API_KEY", "local")

# Model configurations
MODELS = {
    "gpt-4o": {
        "name": "gpt-5-chat",
        "provider": "azure_openai",
        "api_key": AZURE_OPENAI_API_KEY,
        "endpoint": AZURE_OPENAI_ENDPOINT
    },
    "gpt-5-chat": {
        "name": "gpt-5-chat",
        "provider": "azure_openai",
        "api_key": AZURE_OPENAI_API_KEY,
        "endpoint": AZURE_OPENAI_ENDPOINT
    },
    "nvidia/nvidia-nemotron-nano-9b-v2": {
        "name": "nvidia/nvidia-nemotron-nano-9b-v2",
        "provider": "nvidia",
        "api_key": NVIDIA_API_KEY,
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions"
    },
    "nvidia/nvidia-nemotron-nano-vlm-12b": {
        "name": "nvidia/nvidia-nemotron-nano-vlm-12b",
        "provider": "nvidia",
        "api_key": NVIDIA_API_KEY,
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions"
    },
    "nvidia/nemotron-nano-12b-v2-vl": {
        "name": "nvidia/nemotron-nano-12b-v2-vl",
        "provider": "nvidia",
        "api_key": NVIDIA_API_KEY,
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions"
    },
    "microsoft/Phi-4-multimodal-instruct": {
        "name": "microsoft/Phi-4-multimodal-instruct",
        "provider": "openai_compatible",
        "api_key": LOCAL_OPENAI_API_KEY,
        "endpoint": PHI4_MULTIMODAL_ENDPOINT
    },
    "microsoft/phi-4": {
        "name": "microsoft/phi-4",
        "provider": "openai_compatible",
        "api_key": LOCAL_OPENAI_API_KEY,
        "endpoint": PHI4_REASONING_ENDPOINT
    }
}


def encode_image_to_base64(image_path: str, max_size: Tuple[int, int] = (512, 512)) -> str:
    """
    Resize and encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimensions (width, height) to resize to
        
    Returns:
        Base64 encoded string of the resized image
    """
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Resize image while maintaining aspect ratio
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save to bytes buffer with compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70, optimize=True)
        buffer.seek(0)

        b64 = base64.b64encode(buffer.read()).decode('utf-8')
        # Guard against oversized images causing token overflow
        if len(b64) > 150_000:
            raise ValueError("Image too large after compression. Please upload a smaller image.")
        return b64


def call_router(messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Call the router endpoint to determine which model to use.
    
    Returns:
        Tuple of (model_name, error_message)
    """
    payload = {
        "messages": messages,
        "stream": False
    }

    for attempt in range(3):
        try:
            response = requests.post(
                ROUTER_ENDPOINT,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json"
                },
                timeout=240
            )
            response.raise_for_status()

            result = response.json()
            model_name = result["choices"][0]["message"]["content"]
            return model_name, None

        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(10)
                continue
            return None, "Router error: timed out (model may be switching; wait for load to finish)."
        except Exception as e:
            return None, f"Router error: {str(e)}"


def strip_images_for_text_models(messages: List[Dict]) -> List[Dict]:
    """Remove image content from messages for text-only models to avoid token overflow."""
    sanitized = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = " ".join(text_parts).strip() or "[image omitted]"
        sanitized.append({**msg, "content": content})
    return sanitized


def _message_text_length(message: Dict) -> int:
    """Estimate text length for context budgeting."""
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                total += len(item.get("text", ""))
        return total
    return 0


def _trim_messages_for_context(messages: List[Dict], max_chars: int = 22000) -> List[Dict]:
    """Trim oldest messages until request fits a conservative context budget."""
    trimmed = list(messages)

    while len(trimmed) > 1 and sum(_message_text_length(m) for m in trimmed) > max_chars:
        trimmed.pop(0)

    if sum(_message_text_length(m) for m in trimmed) > max_chars and trimmed:
        last = dict(trimmed[-1])
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = content[-max_chars:]
        elif isinstance(content, list):
            rebuilt = []
            budget = max_chars
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    keep = text[-budget:] if budget > 0 else ""
                    rebuilt.append({**item, "text": keep})
                    budget = max(0, budget - len(keep))
                else:
                    rebuilt.append(item)
            last["content"] = rebuilt
        trimmed[-1] = last

    return trimmed


def call_model_azure_openai(model_config: Dict, messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Azure OpenAI API.
    
    Returns:
        Tuple of (response_text, error_message)
    """
    try:
        client = AzureOpenAI(
            azure_endpoint=model_config["endpoint"],
            api_key=model_config["api_key"],
            api_version="2025-01-01-preview"
        )
        
        response = client.chat.completions.create(
            model=model_config["name"],
            messages=messages,
            max_tokens=4096
        )
        
        return response.choices[0].message.content, None
        
    except Exception as e:
        return None, f"Azure OpenAI API error: {str(e)}"


def call_model_nvidia(model_config: Dict, messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Call NVIDIA Build API.
    
    Returns:
        Tuple of (response_text, error_message)
    """
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=model_config["api_key"]
        )
        
        response = client.chat.completions.create(
            model=model_config["name"],
            messages=messages,
            max_tokens=4096
        )
        
        return response.choices[0].message.content, None
        
    except Exception as e:
        return None, f"NVIDIA API error: {str(e)}"


def call_model_openai_compatible(model_config: Dict, messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Call a local or OpenAI-compatible endpoint (e.g., vLLM).
    
    Returns:
        Tuple of (response_text, error_message)
    """
    client = OpenAI(
        base_url=model_config["endpoint"],
        api_key=model_config.get("api_key") or "local"
    )

    if model_config.get("name") == "microsoft/phi-4":
        messages = strip_images_for_text_models(messages)

    messages = _trim_messages_for_context(messages)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_config["name"],
                messages=messages,
                max_tokens=4096,
                timeout=180
            )

            return response.choices[0].message.content, None

        except Exception as e:
            err_text = str(e)
            if "maximum context length" in err_text.lower() and attempt < 2:
                messages = _trim_messages_for_context(messages, max_chars=12000)
                time.sleep(1)
                continue
            if attempt < 2:
                time.sleep(10)
                continue
            return None, f"OpenAI-compatible API error: {str(e)}"


def call_model(model_name: str, messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Call the specified model with the given messages.
    
    Returns:
        Tuple of (response_text, error_message)
    """
    if model_name not in MODELS:
        return None, f"Unknown model: {model_name}"
    
    model_config = MODELS[model_name]

    if model_config["provider"] in {"azure_openai", "nvidia"} and not model_config["api_key"]:
        return None, f"API key not configured for {model_name}"
    
    if model_config["provider"] == "azure_openai":
        return call_model_azure_openai(model_config, messages)
    elif model_config["provider"] == "nvidia":
        return call_model_nvidia(model_config, messages)
    elif model_config["provider"] == "openai_compatible":
        return call_model_openai_compatible(model_config, messages)
    else:
        return None, f"Unknown provider: {model_config['provider']}"


def format_message_with_image(text: str, image_path: Optional[str]) -> Dict:
    """Format a user message with optional image."""
    if image_path:
        # For multimodal messages, use content array format
        base64_image = encode_image_to_base64(image_path)
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    else:
        return {
            "role": "user",
            "content": text
        }


def get_actual_model_name(router_model_name: str) -> str:
    """Get the actual model name that will be called."""
    if router_model_name in MODELS:
        return MODELS[router_model_name]["name"]
    return router_model_name


def format_user_message_for_display(text: str, image_path: Optional[str]) -> str:
    """Format user message for Gradio display with image."""
    if image_path:
        # Convert image to base64 for HTML embedding
        try:
            img_base64 = encode_image_to_base64(image_path)
            return f'{text}<br/><img src="data:image/jpeg;base64,{img_base64}" style="max-width: 400px; max-height: 400px; margin-top: 10px; border-radius: 8px;">'
        except Exception:
            return f"{text}\n[Image: {image_path}]"
    return text


def limit_conversation_history(messages: List[Dict], max_exchanges: int = 5) -> List[Dict]:
    """
    Limit conversation history to the most recent N user/assistant exchanges.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_exchanges: Maximum number of user/assistant exchanges to keep
        
    Returns:
        Limited list of messages
    """
    if len(messages) <= max_exchanges * 2:
        return messages
    
    # Keep the most recent messages (each exchange is 2 messages: user + assistant)
    # Always include the last message which is the current user message
    return messages[-(max_exchanges * 2):]


def chat(message: str, image: Optional[str], history: List[List]) -> Tuple[List[List], str]:
    """
    Process a chat message through the router and model.
    
    Args:
        message: User's text message
        image: Path to uploaded image (optional)
        history: Chat history (list of [user_content, assistant_msg] pairs)
                 user_content can be a string or a dict with text and image
        
    Returns:
        Tuple of (updated_history, status_message)
    """
    if not message and not image:
        return history, "⚠️ Please enter a message or upload an image."
    
    # Build messages list from history for API calls
    # Strip the model prefix from previous assistant messages and reconstruct full context
    messages = []
    for user_content, assistant_msg in history:
        # Handle user messages - could be string or dict with image info
        if isinstance(user_content, dict) and "image" in user_content:
            # Reconstruct multimodal message from stored history
            try:
                user_api_msg = format_message_with_image(
                    user_content["text"],
                    user_content["image"]
                )
                messages.append(user_api_msg)
            except Exception as e:
                history.append([user_content, f"❌ {str(e)}"])
                return history, str(e)
        else:
            # Text-only message
            text = user_content if isinstance(user_content, str) else user_content.get("text", "")
            messages.append({"role": "user", "content": text})
        
        if assistant_msg:
            # Remove the "**Response from model:** " prefix if present
            clean_msg = assistant_msg.split("\n\n", 1)[-1] if "\n\n" in assistant_msg else assistant_msg
            messages.append({"role": "assistant", "content": clean_msg})
    
    # Add current message
    try:
        current_message = format_message_with_image(message or "What's in this image?", image)
    except Exception as e:
        history.append([message or "(image)", f"❌ {str(e)}"])
        return history, str(e)
    messages.append(current_message)
    
    # Limit conversation history to prevent token overflow (stricter for multimodal)
    has_image = any(isinstance(m.get("content"), list) for m in messages)
    messages = limit_conversation_history(messages, max_exchanges=2 if has_image else 3)
    
    # Step 1: Call router to determine model
    status = "🔍 Routing request..."
    model_name, error = call_router(messages)
    
    if error:
        # Add user message to history with image
        user_text = message or "What's in this image?"
        if image:
            # Store both text and image reference for future API calls
            user_storage = {
                "text": user_text,
                "image": image
            }
        else:
            user_storage = user_text
        
        history.append([user_storage, f"❌ {error}"])
        return history, error
    
    # Get the actual model name that will be called
    actual_model = get_actual_model_name(model_name)
    status = f"✅ Router selected model, getting response..."
    
    # Step 2: Call the selected model
    response, error = call_model(model_name, messages)
    
    if error:
        # Add user message to history with image
        user_text = message or "What's in this image?"
        if image:
            user_storage = {
                "text": user_text,
                "image": image
            }
        else:
            user_storage = user_text
        
        history.append([user_storage, f"❌ {error}"])
        return history, error
    
    # Add to history with model name prefix and image thumbnail
    user_text = message or "What's in this image?"
    if image:
        # Store structured data for API calls
        user_storage = {
            "text": user_text,
            "image": image
        }
    else:
        user_storage = user_text
    
    # Format response with model name in bold
    formatted_response = f"**Response from {actual_model}:**\n\n{response}"
    
    history.append([user_storage, formatted_response])
    
    status = "Complete"
    
    return history, status


def create_demo():
    """Create and configure the Gradio interface."""
    
    # NVIDIA theme with brand colors
    nvidia_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
    ).set(
        button_primary_background_fill="#76B900",
        button_primary_background_fill_hover="#5f9400",
        button_primary_text_color="white",
        block_title_text_color="#76B900",
        block_label_text_color="#76B900",
        input_background_fill="#1a1a1a",
        body_background_fill="#0d0d0d",
        body_text_color="#ffffff",
    )
    
    with gr.Blocks(title="I got Phi on it GB10 Demo (forked from https://github.com/NVIDIA-AI-Blueprints/llm-router)", theme=nvidia_theme, css="""
        #chatbot {
            background-color: #1a1a1a;
        }
        /* Style chat messages for readability */
        .message {
            border-radius: 8px;
        }
        .message.user,
        .message.bot,
        .user-row,
        .bot-row {
            background-color: #2a2a2a !important;
            color: #ffffff !important;
            border: 1px solid #404040 !important;
        }
        .message p,
        .message span,
        .message div {
            color: #ffffff !important;
        }
        /* User messages */
        [data-testid="user"] {
            background-color: #2a2a2a !important;
        }
        /* Bot messages */
        [data-testid="bot"] {
            background-color: #1f1f1f !important;
        }
        /* Remove white border around input area */
        .input-row {
            border: none !important;
            background: transparent !important;
        }
        /* Style text input */
        textarea {
            border: 1px solid #333 !important;
            background-color: #1a1a1a !important;
        }
        /* Style image upload */
        .image-container {
            border: 1px solid #333 !important;
            background-color: #1a1a1a !important;
        }
        /* Remove ALL dividing lines and borders inside upload box */
        .image-container hr,
        .image-container .divider,
        .image-container [class*="border"],
        .image-container [class*="divider"] {
            display: none !important;
            border: none !important;
            visibility: hidden !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        /* Target specific Gradio upload divider */
        [data-testid="image"] hr,
        [data-testid="image"] [class*="or"],
        .image-container .or-text {
            display: none !important;
        }
        /* Hide ALL icons and buttons in image upload area */
        .image-container button[title],
        .image-container button svg,
        .image-container .icon-buttons,
        .image-container .download-button,
        .image-container .share-button,
        .image-container .clear-button,
        .image-container footer,
        .image-container footer button {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
        }
        /* Make upload area smaller and text fit */
        .image-container .wrap {
            min-height: 80px !important;
            max-height: 80px !important;
        }
        .image-container {
            height: 80px !important;
        }
        .image-container .upload-text,
        .image-container span,
        .image-container .or {
            font-size: 0.7rem !important;
            line-height: 1.1 !important;
        }
    """) as demo:
        gr.Markdown("# I got Phi on it GB10 Demo (forked from https://github.com/NVIDIA-AI-Blueprints/llm-router)")
        gr.Markdown("⚠️ **Model switching notice**: Image requests will switch to the Phi-4 multimodal server, and text requests switch back to Phi-4 reasoning. This can take a few minutes while containers stop/start. For seamless use, consider adding another GB10 to run both models simultaneously.")
        
        chatbot = gr.Chatbot(
            label="I got Phi on it GB10 Demo (forked from https://github.com/NVIDIA-AI-Blueprints/llm-router)",
            height=600,
            show_label=False,
            elem_id="chatbot"
        )
        
        with gr.Row(elem_classes="input-row"):
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                scale=3,
                lines=2,
                show_label=False,
                container=False
            )
            image = gr.Image(
                label="",
                type="filepath",
                scale=1,
                height=80,
                show_label=False,
                show_download_button=False,
                show_share_button=False,
                sources=["upload"],
                container=True,
                elem_classes="image-container",
                interactive=True
            )
        
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Chat")
        
        status = gr.Markdown("")
        
        # Event handlers
        def submit_message(message, image, history):
            new_history, status_msg = chat(message, image, history)
            # Convert storage format to display format for Gradio
            display_history = []
            for user_content, assistant_msg in new_history:
                if isinstance(user_content, dict) and "image" in user_content:
                    # Format with image for display
                    user_display = format_user_message_for_display(
                        user_content["text"],
                        user_content["image"]
                    )
                else:
                    user_display = user_content
                display_history.append([user_display, assistant_msg])
            return display_history, status_msg
        
        def clear_chat():
            return [], ""
        
        submit_btn.click(
            fn=submit_message,
            inputs=[msg, image, chatbot],
            outputs=[chatbot, status]
        ).then(
            fn=lambda: ("", None),  # Clear message and image after sending
            outputs=[msg, image]
        )
        
        msg.submit(
            fn=submit_message,
            inputs=[msg, image, chatbot],
            outputs=[chatbot, status]
        ).then(
            fn=lambda: ("", None),
            outputs=[msg, image]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, status]
        )
    
    return demo


if __name__ == "__main__":
    # Check configuration
    print("🔧 Checking configuration...")
    print(f"Router Endpoint: {ROUTER_ENDPOINT}")
    print(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT or '✗ Missing'}")
    print(f"Azure OpenAI API Key: {'✓ Configured' if AZURE_OPENAI_API_KEY else '✗ Missing'}")
    print(f"NVIDIA API Key: {'✓ Configured' if NVIDIA_API_KEY else '✗ Missing'}")
    print("\n" + "="*50 + "\n")
    
    # Launch the app
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

