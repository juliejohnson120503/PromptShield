"""
PromptShield AI — Core Backend Package
A Context-Aware Security Layer for Safe, Responsible, and Enterprise-Ready AI Systems.
"""

from backend.models import (
    EntityType,
    DetectedEntity,
    BaselineMaskResult,
    TaskType,
    RoleCategory,
    ContextualRole,
    ContextAnalysisResult,
)
from backend.detector import SensitiveDataDetector
from backend.regex_detector import RegexDetector, is_luhn_valid, detect_card_brand
from backend.presidio_detector import PresidioDetector
from backend.spacy_detector import SpacyNERDetector
from backend.entity_fusion import EntityFusionEngine
from backend.mapping_store import MappingStore
from backend.baseline_masker import BaselineMasker
from backend.task_classifier import TaskClassifier
from backend.context_analyzer import ContextAnalyzer
from backend.core import PromptShieldCore
from backend.normalization import normalize_entity_value

__version__ = "0.3.0"

__all__ = [
    # Models
    "EntityType",
    "DetectedEntity",
    "BaselineMaskResult",
    "TaskType",
    "RoleCategory",
    "ContextualRole",
    "ContextAnalysisResult",
    # Detection
    "SensitiveDataDetector",
    "RegexDetector",
    "PresidioDetector",
    "SpacyNERDetector",
    "EntityFusionEngine",
    # Utilities
    "is_luhn_valid",
    "detect_card_brand",
    # Pipeline
    "MappingStore",
    "BaselineMasker",
    "TaskClassifier",
    "ContextAnalyzer",
    "PromptShieldCore",
    "normalize_entity_value",
]


