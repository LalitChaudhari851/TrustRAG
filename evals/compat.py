"""
In-repo compatibility shims for third-party libraries (e.g. Ragas) 
that expect legacy module namespaces in langchain_community without modifying .venv files.
"""

import sys
import types

def apply_compat_shims():
    """Inject dummy modules for deprecated VertexAI paths in langchain_community."""
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        m = types.ModuleType("langchain_community.chat_models.vertexai")
        m.ChatVertexAI = None
        sys.modules["langchain_community.chat_models.vertexai"] = m
    
    if "langchain_community.llms.vertexai" not in sys.modules:
        m2 = types.ModuleType("langchain_community.llms.vertexai")
        m2.VertexAI = None
        sys.modules["langchain_community.llms.vertexai"] = m2

apply_compat_shims()
