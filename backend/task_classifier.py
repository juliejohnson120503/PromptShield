"""
Task Classification Module for PromptShield AI (Phase 2).
Identifies the user's task intent to provide context for risk and policy decisions.
Supports 8 canonical task categories:
- GENERAL_QA
- EMAIL_GENERATION
- CODE_GENERATION
- SUMMARIZATION
- TRANSLATION
- DATA_ANALYSIS
- DOCUMENT_GENERATION
- GENERAL_CHAT
"""

import re
from typing import Tuple, List, Dict
from backend.models import TaskType


# Intent patterns and keyword indicators for each task category
TASK_PATTERNS: Dict[TaskType, List[re.Pattern]] = {
    TaskType.EMAIL_GENERATION: [
        re.compile(r"(?i)\b(?:write|draft|compose|send|create)\s+(?:a\s+|an\s+)?(?:professional\s+|formal\s+|follow-up\s+|internship\s+|job\s+)?email\b"),
        re.compile(r"(?i)\b(?:email\s+(?:to|regarding|about)|cover\s+letter|cold\s+email)\b"),
        re.compile(r"(?i)\b(?:subject\s*:\s*|dear\s+hiring\s+manager|to\s+whom\s+it\s+may\s+concern)\b"),
    ],
    TaskType.CODE_GENERATION: [
        re.compile(r"(?i)\b(?:write|generate|implement|debug|refactor|fix)\s+(?:a\s+|an\s+|the\s+|this\s+)?(?:code|script|function|class|method|program|algorithm|query|api|component|bug|error|issue|app)\b"),
        re.compile(r"(?i)\b(?:python|javascript|typescript|java|c\+\+|golang|rust|sql|html|css|react|vue|angular|node|django|fastapi)\s+(?:code|script|function|snippet|component|app)\b"),
        re.compile(r"(?i)\b(?:fix\s+(?:the\s+|this\s+)?(?:state\s+update\s+)?bug|state\s+update\s+bug)\b"),
        re.compile(r"\b(?:def\s+\w+\(|function\s+\w+\(|SELECT\s+.+\s+FROM|public\s+class\s+\w+)\b"),
    ],
    TaskType.SUMMARIZATION: [
        re.compile(r"(?i)\b(?:summarize|summary\s+of|tldr|give\s+me\s+a\s+summary|brief\s+overview|condense|key\s+takeaways|bullet\s+points\s+of)\b"),
        re.compile(r"(?i)\b(?:in\s+short|in\s+a\s+nutshell|summarise)\b"),
    ],
    TaskType.TRANSLATION: [
        re.compile(r"(?i)\b(?:translate|translation\s+of)\b"),
        re.compile(r"(?i)\b(?:into|to|in)\s+(?:spanish|french|german|hindi|chinese|japanese|russian|italian|portuguese|arabic|english)\b"),
    ],
    TaskType.DATA_ANALYSIS: [
        re.compile(r"(?i)\b(?:analyze|analyse|evaluate)\s+(?:the\s+)?(?:data|dataset|numbers|metrics|statistics|table|chart|trend)\b"),
        re.compile(r"(?i)\b(?:calculate\s+(?:the\s+)?(?:mean|median|average|std|correlation)|regression|data\s+analysis)\b"),
    ],
    TaskType.DOCUMENT_GENERATION: [
        re.compile(r"(?i)\b(?:write|draft|create|generate)\s+(?:a\s+|an\s+)?(?:essay|resume|cv|report|contract|agreement|article|blog\s+post|proposal|story)\b"),
    ],
    TaskType.GENERAL_QA: [
        re.compile(r"(?i)\b(?:what\s+is|who\s+is|who\s+was|where\s+is|when\s+did|why\s+did|why\s+is|how\s+does|how\s+do|how\s+can|how\s+to)\b"),
        re.compile(r"(?i)\b(?:tell\s+me\s+about|explain|describe|give\s+details\s+about|what\s+are|meaning\s+of|capital\s+of|history\s+of)\b"),
        re.compile(r"(?i)\b(?:difference\s+between|compare\s+.+\s+and)\b"),
    ],
    TaskType.GENERAL_CHAT: [
        re.compile(r"(?i)\b(?:hello|hi|hey|good\s+(?:morning|afternoon|evening)|how\s+are\s+you|nice\s+to\s+meet\s+you|greetings)\b"),
        re.compile(r"(?i)\b(?:my\s+name\s+is|i\s+am)\b"),
    ],
}


class TaskClassifier:
    """
    Classifies prompt task intent using transparent, explainable heuristics.
    Pre-configured for future semantic embedding enhancement.
    """

    def classify(self, prompt: str) -> Tuple[TaskType, float, List[str]]:
        """
        Evaluate prompt and return (TaskType, confidence, matching_cues).
        Falls back to GENERAL_QA or GENERAL_CHAT if ambiguous.
        """
        cleaned = prompt.strip()
        if not cleaned:
            return TaskType.GENERAL_CHAT, 1.0, ["empty_prompt"]

        scores: Dict[TaskType, int] = {t: 0 for t in TaskType}
        detected_cues: Dict[TaskType, List[str]] = {t: [] for t in TaskType}

        # Check explicit regex patterns with priority weighting
        for task_type, patterns in TASK_PATTERNS.items():
            for pattern in patterns:
                matches = pattern.findall(cleaned)
                if matches:
                    weight = 3 if task_type in (
                        TaskType.EMAIL_GENERATION,
                        TaskType.CODE_GENERATION,
                        TaskType.TRANSLATION,
                        TaskType.SUMMARIZATION,
                    ) else 2
                    scores[task_type] += weight * len(matches)
                    detected_cues[task_type].append(pattern.pattern)

        # Disambiguate top task
        best_task = max(scores, key=lambda t: scores[t])
        best_score = scores[best_task]

        if best_score > 0:
            confidence = min(0.95, 0.65 + 0.1 * best_score)
            return best_task, confidence, detected_cues[best_task]

        # If question mark is present, default to GENERAL_QA
        if "?" in cleaned:
            return TaskType.GENERAL_QA, 0.70, ["question_mark_structure"]

        # Default fallback
        return TaskType.GENERAL_CHAT, 0.60, ["fallback_conversational"]
