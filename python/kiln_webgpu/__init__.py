"""Kiln for Python: models on the GPU (WebGPU) or the CPU, from their original safetensors or GGUF
files, through the same pipelines as the JavaScript library. The models run in a Node.js process this
package starts.

    from kiln_webgpu import pipeline

    embed = pipeline("feature-extraction", "sentence-transformers/all-MiniLM-L6-v2")
    vectors = embed(["Hello world", "Hi"], pooling="mean", normalize=True)
"""
from ._bridge import install_gpu
from ._pipeline import (Pipeline, calculator_skill, date_time_skill, invoke, pipeline, run_tools, runtime, set_access_token, set_env, shutdown,
                        skill_instructions, skills, web_page_skill, wikipedia_skill)
from ._types import KilnError, RawImage, Tensor, write_wav

__version__ = "0.3.0"
__all__ = ["pipeline", "Pipeline", "Tensor", "RawImage", "KilnError", "install_gpu", "invoke", "run_tools", "runtime", "set_access_token", "set_env", "shutdown", "write_wav",
           "skills", "calculator_skill", "date_time_skill", "wikipedia_skill", "web_page_skill", "skill_instructions"]
