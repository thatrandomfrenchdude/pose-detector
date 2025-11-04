import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import onnxruntime as ort
import numpy as np

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Model loader for ONNX models with CPU/NPU support.
    Handles model loading, validation, and provider selection.
    """
    
    def __init__(self, model_path: str, preferred_provider: str = "cpu"):
        """
        Initialize the ModelLoader.
        
        Args:
            model_path: Path to the ONNX model file
            preferred_provider: Preferred execution provider ("cpu", "npu", "gpu")
        """
        self.model_path = Path(model_path)
        self.preferred_provider = preferred_provider.lower()
        self.session: Optional[ort.InferenceSession] = None
        self.input_details = None
        self.output_details = None
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
    def load_model(self) -> ort.InferenceSession:
        """
        Load the ONNX model with appropriate execution provider.
        
        Returns:
            ort.InferenceSession: Loaded ONNX inference session
        """
        # Get available providers
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        # Select execution provider based on preference and availability
        providers = self._select_providers(available_providers)
        
        try:
            # Create session options for better performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Enable QNN context binary caching for faster startup
            if any(isinstance(p, tuple) and p[0] == "QNNExecutionProvider" for p in providers):
                # Enable context caching to avoid recompilation
                sess_options.add_session_config_entry("ep.context_enable", "1")
                
                # Set context file path for caching
                context_file = str(self.model_path.with_suffix('.ctx.onnx'))
                sess_options.add_session_config_entry("ep.context_file_path", context_file)
                
                # Embed context binary for easier deployment
                sess_options.add_session_config_entry("ep.context_embed_mode", "1")
                
                logger.info(f"QNN context caching enabled: {context_file}")
            
            # For QNN EP, optionally disable CPU fallback for pure NPU execution
            qnn_in_providers = any(isinstance(p, tuple) and p[0] == "QNNExecutionProvider" for p in providers)
            if qnn_in_providers:
                # Only disable CPU fallback if CPU provider is not explicitly included
                cpu_provider_included = any(
                    (isinstance(p, str) and p == "CPUExecutionProvider") or 
                    (isinstance(p, tuple) and p[0] == "CPUExecutionProvider") 
                    for p in providers
                )
                
                if not cpu_provider_included:
                    sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                    logger.info("Configured session for QNN EP without CPU fallback")
                else:
                    logger.info("Configured session for QNN EP with CPU fallback enabled")
            
            # Separate providers and provider_options for session creation
            provider_list = []
            provider_options_list = []
            
            for provider in providers:
                if isinstance(provider, tuple):
                    provider_list.append(provider[0])
                    provider_options_list.append(provider[1])
                else:
                    provider_list.append(provider)
                    provider_options_list.append({})
            
            # Load the model
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=provider_list,
                provider_options=provider_options_list
            )
            
            # Get input/output details
            self.input_details = self.session.get_inputs()
            self.output_details = self.session.get_outputs()
            
            logger.info(f"Model loaded successfully with provider: {self.session.get_providers()[0]}")
            logger.info(f"Input shape: {self.input_details[0].shape}")
            logger.info(f"Output shape: {self.output_details[0].shape}")
            
            return self.session
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _select_providers(self, available_providers: list) -> list:
        """
        Select the best available execution provider.
        
        Args:
            available_providers: List of available ONNX providers
            
        Returns:
            list: Ordered list of providers to try
        """
        provider_mapping = {
            "npu": ["QNNExecutionProvider"],
            "gpu": ["CUDAExecutionProvider", "DirectMLExecutionProvider"], 
            "cpu": ["CPUExecutionProvider"]
        }
        
        # QNN provider options for NPU acceleration
        qnn_provider_options = {
            "backend_path": "QnnHtp.dll",  # HTP backend for NPU
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_fp16_precision": "1",  # Enable FP16 for better NPU performance
            "qnn_context_priority": "high",
            "profiling_level": "basic"  # Enable basic profiling
        }
        
        selected_providers = []
        
        # Add preferred provider if available
        if self.preferred_provider in provider_mapping:
            for provider in provider_mapping[self.preferred_provider]:
                if provider in available_providers:
                    if provider == "QNNExecutionProvider":
                        selected_providers.append((provider, qnn_provider_options))
                        # For NPU preference, try NPU-only first, then add CPU fallback
                        if self.preferred_provider == "npu":
                            logger.info("NPU-first configuration: QNN EP will attempt NPU execution")
                    else:
                        selected_providers.append(provider)
                    break
        
        # Add CPU as fallback only if not NPU-only mode
        should_add_cpu_fallback = True
        if self.preferred_provider == "npu":
            # Check if QNN was successfully added
            qnn_added = any(isinstance(p, tuple) and p[0] == "QNNExecutionProvider" for p in selected_providers)
            if qnn_added:
                # For testing, we can try NPU-only mode first
                should_add_cpu_fallback = True  # Change to False for pure NPU mode
        
        if should_add_cpu_fallback and "CPUExecutionProvider" not in [p if isinstance(p, str) else p[0] for p in selected_providers]:
            selected_providers.append("CPUExecutionProvider")
        
        logger.info(f"Selected providers: {[p if isinstance(p, str) else p[0] for p in selected_providers]}")
        return selected_providers
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get the expected input shape."""
        if self.input_details:
            return tuple(self.input_details[0].shape)
        return None
    
    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Get the expected output shape."""
        if self.output_details:
            return tuple(self.output_details[0].shape)
        return None
    
    @property
    def input_name(self) -> str:
        """Get the input tensor name."""
        if self.input_details:
            return self.input_details[0].name
        return None
    
    @property
    def output_name(self) -> str:
        """Get the output tensor name."""
        if self.output_details:
            return self.output_details[0].name
        return None
    
    def get_session_info(self) -> dict:
        """
        Get detailed information about the loaded session.
        
        Returns:
            dict: Session information including providers, shapes, etc.
        """
        if not self.session:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "providers": self.session.get_providers(),
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_type": self.input_details[0].type if self.input_details else None,
            "output_type": self.output_details[0].type if self.output_details else None,
        }