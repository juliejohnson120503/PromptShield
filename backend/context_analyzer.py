"""
Contextual Role Analyzer Module for PromptShield AI (Phase 2).
Analyzes the semantic relationship between detected entities and the user's prompt.
Evaluates:
- Personal vs. Public Information (e.g. "My name is Julie" vs "Tell me about Elon Musk")
- First-party vs. Third-party possession
- Task relevance (e.g., recipient organization in job application email)
- Technical credentials vs. General knowledge
"""

import re
from typing import List, Dict, Set
from backend.models import (
    EntityType,
    DetectedEntity,
    TaskType,
    RoleCategory,
    ContextualRole,
    ContextAnalysisResult,
)


# Known Public Personalities / Celebrities / Historical Figures
KNOWN_PUBLIC_FIGURES: Set[str] = {
    "elon musk", "bill gates", "steve jobs", "albert einstein",
    "sundar pichai", "satya nadella", "sam altman", "jeff bezos",
    "mark zuckerberg", "barack obama", "narendra modi", "alan turing",
    "marie curie", "isaac newton", "charles babbage", "ada lovelace",
    "geoffrey hinton", "yann lecun", "yoshua bengio", "andrew ng",
    "warren buffett", "tim cook", "jensen huang", "linus torvalds"
}

# Known Public Organizations
KNOWN_PUBLIC_ORGS: Set[str] = {
    "google", "microsoft", "apple", "amazon", "meta", "facebook",
    "openai", "anthropic", "tesla", "spacex", "nasa", "ibm", "nvidia",
    "intel", "cisco", "oracle", "united nations", "who", "tcs", "infosys",
    "wipro", "accenture", "harvard university", "mit", "stanford university"
}

# First-Party Possessive & Personal Identifiers Cues
FIRST_PARTY_CUES = re.compile(
    r"(?i)\b(?:my\s+name\s+is|my\s+email|my\s+phone|my\s+number|my\s+address|my\s+card|"
    r"my\s+password|my\s+key|i\s+am|i'm|call\s+me|contact\s+me|reach\s+me|i\s+live\s+in|"
    r"myself|registered\s+to\s+me|my\s+application)\b"
)

# Inquisitive / Public Inquiry Cues
PUBLIC_INQUIRY_CUES = re.compile(
    r"(?i)\b(?:tell\s+me\s+about|who\s+is|who\s+was|what\s+is|explain|describe|biography\s+of|"
    r"history\s+of|information\s+on|details\s+about|search\s+for|background\s+of|capital\s+of)\b"
)

# Task Relationship Cues (e.g. target company in outreach)
TASK_RELEVANCE_CUES = re.compile(
    r"(?i)\b(?:email\s+to|application\s+(?:to|for)|interview\s+with|working\s+at|"
    r"contract\s+with|regarding\s+my\s+application\s+to|cover\s+letter\s+for)\b"
)


class ContextAnalyzer:
    """
    Evaluates contextual roles of detected entities given the overall prompt and task type.
    Produces explainable ContextualRole objects for downstream policy assessment.
    """

    def analyze_entity_context(
        self,
        entity: DetectedEntity,
        prompt: str,
        task_type: TaskType,
    ) -> ContextualRole:
        """
        Evaluate single entity's context in the prompt.
        """
        lower_text = entity.text.lower().strip()
        normalized_lower = entity.normalized_value.lower().strip()

        # Extract local window of context (up to 40 chars before and after entity)
        start_win = max(0, entity.start - 40)
        end_win = min(len(prompt), entity.end + 40)
        pre_context = prompt[start_win:entity.start]
        local_context = prompt[start_win:end_win]

        # -----------------------------------------------------------------
        # 1. TECHNICAL CREDENTIALS (API_KEY, PASSWORD, CREDIT_CARD)
        # -----------------------------------------------------------------
        if entity.entity_type in (EntityType.API_KEY, EntityType.PASSWORD, EntityType.CREDIT_CARD):
            return ContextualRole(
                entity_text=entity.text,
                entity_type=entity.entity_type,
                role_category=RoleCategory.TECHNICAL_CREDENTIAL,
                is_first_party=True,
                is_public_knowledge=False,
                is_task_relevant=False,
                context_cue=f"Strict technical credential of type {entity.entity_type.value}",
                confidence=0.98,
            )

        # -----------------------------------------------------------------
        # 2. CONTACT PII (EMAIL, PHONE)
        # -----------------------------------------------------------------
        if entity.entity_type in (EntityType.EMAIL, EntityType.PHONE):
            is_first_party = bool(FIRST_PARTY_CUES.search(local_context)) or "my" in pre_context.lower()
            return ContextualRole(
                entity_text=entity.text,
                entity_type=entity.entity_type,
                role_category=RoleCategory.PERSONAL_IDENTIFIER,
                is_first_party=is_first_party,
                is_public_knowledge=False,
                is_task_relevant=(task_type == TaskType.EMAIL_GENERATION),
                context_cue="Personal contact PII detected",
                confidence=0.95,
            )

        # -----------------------------------------------------------------
        # 3. PERSON ENTITY EVALUATION (Julie vs Elon Musk)
        # -----------------------------------------------------------------
        if entity.entity_type == EntityType.PERSON:
            # Check if entity is a recognized public figure
            is_known_public = (
                lower_text in KNOWN_PUBLIC_FIGURES or normalized_lower in KNOWN_PUBLIC_FIGURES
            )

            # Check for first-party self-identification cues ("My name is Julie", "I am...")
            is_self_ident = bool(FIRST_PARTY_CUES.search(pre_context)) or bool(
                re.search(r"(?i)\b(?:my\s+name\s+is|i\s+am|call\s+me)\s+$", pre_context.strip())
            )

            # Check for inquisitive public question cues ("Tell me about Elon Musk", "Who is...")
            is_inquiry = bool(PUBLIC_INQUIRY_CUES.search(prompt)) or task_type == TaskType.GENERAL_QA

            if is_known_public or (is_inquiry and not is_self_ident):
                return ContextualRole(
                    entity_text=entity.text,
                    entity_type=EntityType.PERSON,
                    role_category=RoleCategory.PUBLIC_FIGURE_OR_FACT,
                    is_first_party=False,
                    is_public_knowledge=True,
                    is_task_relevant=True,
                    context_cue="Public figure referenced as subject of inquiry",
                    confidence=0.92,
                )

            # Otherwise, personal name (e.g. "Julie", "John Mathew")
            return ContextualRole(
                entity_text=entity.text,
                entity_type=EntityType.PERSON,
                role_category=RoleCategory.PERSONAL_IDENTIFIER,
                is_first_party=is_self_ident,
                is_public_knowledge=False,
                is_task_relevant=(task_type == TaskType.EMAIL_GENERATION),
                context_cue="Personal name associated with user or individual",
                confidence=0.90,
            )

        # -----------------------------------------------------------------
        # 4. ORGANIZATION EVALUATION
        # -----------------------------------------------------------------
        if entity.entity_type == EntityType.ORGANIZATION:
            is_known_org = (
                lower_text in KNOWN_PUBLIC_ORGS or normalized_lower in KNOWN_PUBLIC_ORGS
            )
            # In an email or document generation task, target company is task-relevant
            is_task_rel = bool(TASK_RELEVANCE_CUES.search(local_context)) or (
                task_type in (TaskType.EMAIL_GENERATION, TaskType.DOCUMENT_GENERATION)
            )

            if is_known_org or (task_type == TaskType.GENERAL_QA and not FIRST_PARTY_CUES.search(pre_context)):
                return ContextualRole(
                    entity_text=entity.text,
                    entity_type=EntityType.ORGANIZATION,
                    role_category=(
                        RoleCategory.TASK_RELEVANT_ENTITY if is_task_rel else RoleCategory.PUBLIC_FIGURE_OR_FACT
                    ),
                    is_first_party=False,
                    is_public_knowledge=True,
                    is_task_relevant=is_task_rel,
                    context_cue="Public commercial or institutional entity",
                    confidence=0.90,
                )

            return ContextualRole(
                entity_text=entity.text,
                entity_type=EntityType.ORGANIZATION,
                role_category=(
                    RoleCategory.TASK_RELEVANT_ENTITY if is_task_rel else RoleCategory.GENERAL_REFERENCE
                ),
                is_first_party=bool(FIRST_PARTY_CUES.search(pre_context)),
                is_public_knowledge=False,
                is_task_relevant=is_task_rel,
                context_cue="Target organizational context for user task",
                confidence=0.88,
            )

        # -----------------------------------------------------------------
        # 5. LOCATION EVALUATION
        # -----------------------------------------------------------------
        if entity.entity_type == EntityType.LOCATION:
            # Check if user says "I live in London" vs "What is the capital of France"
            is_personal_residence = bool(
                re.search(r"(?i)\b(?:i\s+live\s+in|my\s+(?:home|house|office)\s+in|based\s+in)\b", pre_context)
            )
            is_qa_fact = (task_type == TaskType.GENERAL_QA) or bool(PUBLIC_INQUIRY_CUES.search(prompt))

            if is_qa_fact and not is_personal_residence:
                return ContextualRole(
                    entity_text=entity.text,
                    entity_type=EntityType.LOCATION,
                    role_category=RoleCategory.PUBLIC_FIGURE_OR_FACT,
                    is_first_party=False,
                    is_public_knowledge=True,
                    is_task_relevant=True,
                    context_cue="Geographical location cited as public knowledge query",
                    confidence=0.92,
                )

            return ContextualRole(
                entity_text=entity.text,
                entity_type=EntityType.LOCATION,
                role_category=(
                    RoleCategory.PERSONAL_IDENTIFIER if is_personal_residence else RoleCategory.GENERAL_REFERENCE
                ),
                is_first_party=is_personal_residence,
                is_public_knowledge=True,
                is_task_relevant=True,
                context_cue="Location reference associated with personal context" if is_personal_residence else "General geographical reference",
                confidence=0.86,
            )

        # -----------------------------------------------------------------
        # 6. DATE EVALUATION
        # -----------------------------------------------------------------
        if entity.entity_type == EntityType.DATE:
            is_historic = (task_type == TaskType.GENERAL_QA)
            return ContextualRole(
                entity_text=entity.text,
                entity_type=EntityType.DATE,
                role_category=RoleCategory.PUBLIC_FIGURE_OR_FACT if is_historic else RoleCategory.GENERAL_REFERENCE,
                is_first_party=False,
                is_public_knowledge=is_historic,
                is_task_relevant=True,
                context_cue="Temporal anchor or date specification",
                confidence=0.85,
            )

        # Fallback for generic entities
        return ContextualRole(
            entity_text=entity.text,
            entity_type=entity.entity_type,
            role_category=RoleCategory.GENERAL_REFERENCE,
            is_first_party=False,
            is_public_knowledge=False,
            is_task_relevant=False,
            context_cue="General entity without explicit personal or public modifier",
            confidence=0.75,
        )

    def analyze(
        self,
        prompt: str,
        entities: List[DetectedEntity],
        task_type: TaskType,
        task_confidence: float,
        task_cues: List[str],
    ) -> ContextAnalysisResult:
        """
        Perform full context understanding across all detected entities.
        """
        contexts: List[ContextualRole] = []
        for ent in entities:
            role = self.analyze_entity_context(ent, prompt, task_type)
            contexts.append(role)

        return ContextAnalysisResult(
            prompt=prompt,
            task_type=task_type,
            task_confidence=task_confidence,
            task_cues=task_cues,
            entity_contexts=contexts,
        )
