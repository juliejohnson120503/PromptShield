"""
Contextual Role Analyzer Module for PromptShield AI (Phase 2).
Analyzes the semantic relationship between detected entities and the user's prompt.
Evaluates:
- Personal vs. Public Information (e.g. "My name is Julie" vs "Tell me about Elon Musk")
- First-party vs. Third-party possession
- Task relevance (e.g., recipient organization in job application email)
- Technical credentials vs. General knowledge

Phase 4 note:
  Character offsets (start, end) and normalized_value from DetectedEntity are
  threaded into every returned ContextualRole so the SemanticMasker can perform
  span-based masking without re-detecting entities.
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from backend.models import (
    EntityType,
    DetectedEntity,
    TaskType,
    RoleCategory,
    ContextualRole,
    ContextAnalysisResult,
    EntityTaskRelation,
    OperationType,
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

# Recovery Question & Authentication Secret Cues
RECOVERY_QUESTION_CUES = re.compile(
    r"(?i)\b(?:mother'?s\s+maiden\s+name|security\s+question(?:\s+answer)?|"
    r"account\s+recovery(?:\s+answer)?|secret\s+question(?:\s+answer)?)\b"
)


def detect_operation(prompt: str) -> OperationType:
    """
    Identify the broad semantic operation requested in the prompt.
    """
    cleaned = prompt.strip()
    if not cleaned:
        return OperationType.UNKNOWN

    if re.search(
        r"(?i)\b(?:strength|entropy|complexity|character\s*count|how\s+many\s+(?:digits?|letters?|chars?|characters?|numbers?|symbols?)|"
        r"length\s+of|count\s+(?:the\s+)?(?:digits?|letters?|chars?|characters?)|"
        r"pattern\s+of|structure\s+of|starts?\s+with|ends?\s+with|contains?\s+(?:any\s+)?(?:digits?|letters?|numbers?|symbols?)|"
        r"is\s+(?:this|it|the|my)?\s*(?:password|string|key|secret|code|token)?\s*(?:strong|weak|secure|safe|complex|compromised))\b",
        cleaned,
    ):
        return OperationType.VALUE_ANALYSIS

    if re.search(
        r"(?i)\b(?:compare|comparison|difference\s+between|which\s+(?:one\s+)?is\s+(?:stronger|better|longer|faster|more\s+secure)|"
        r"same\s+as|versus|vs\.?)\b",
        cleaned,
    ):
        return OperationType.COMPARISON

    if re.search(
        r"(?i)\b(?:calculate|compute|derive|what\s+is\s+(?:my|the)?\s*age|how\s+old(?:\s+am\s+i|\s+is)?|"
        r"age\s+of|sum\s+of|difference\s+in\s+years|years\s+between|days\s+until)\b",
        cleaned,
    ):
        return OperationType.CALCULATION

    if re.search(
        r"(?i)\b(?:validate|verify|check\s+format|is\s+(?:this|it|the|my)?\s*(?:a\s+|an\s+)?(?:valid|invalid|private|public|reserved|routable|authorized|well-formed|malformed))\b",
        cleaned,
    ):
        return OperationType.VALIDATION

    if re.search(
        r"(?i)\b(?:rewrite|rephrase|paraphrase|reword|proofread|sanitize|clean\s+up|edit\s+this)\b",
        cleaned,
    ):
        return OperationType.TRANSFORMATION

    if re.search(
        r"(?i)\b(?:summarize|summary\s+of|tldr|brief\s+overview|condense|key\s+takeaways)\b",
        cleaned,
    ):
        return OperationType.SUMMARIZATION

    if re.search(
        r"(?i)\b(?:write|draft|compose|create)\s+(?:a\s+|an\s+)?(?:email|letter|application|essay|report|cover\s+letter|memo|story)\b",
        cleaned,
    ):
        return OperationType.GENERATION

    if re.search(
        r"(?i)\b(?:explain|tell\s+me\s+about|what\s+is|who\s+is|describe|definition\s+of)\b",
        cleaned,
    ):
        return OperationType.GENERAL_INFORMATION

    return OperationType.UNKNOWN


# Linguistic Directive Patterns (Phase 2 & Phase 3)
INCLUSION_DIRECTIVE_PATTERN = re.compile(
    r"(?i)\b(?:must\s+mention|please\s+mention|mention(?:ing|ed|s)?|must\s+include|please\s+include|include|including|includes|refer(?:ring)?\s+to|state|stating|cite|citing|must\s+contain|should\s+mention|use\s+in\s+the\s+report)\s+([^.?!;\n]+)"
)
EXCLUSION_DIRECTIVE_PATTERN = re.compile(
    r"(?i)\b(?:(?:you\s+)?(?:must|may|might|should|shall|can|cannot|could|do|will)\s+not\s+(?:expose|reveal|disclose|leak|display|include|mention|show|share)|"
    r"don't\s+(?:expose|reveal|disclose|leak|display|include|mention|show|share)|"
    r"never\s+(?:expose|reveal|disclose|leak|display|include|mention|show|share)|"
    r"not\s+(?:expose|reveal|disclose|leak|display|include|mention|show|share)|"
    r"hide|mask|redact|keep\s+(?:private|confidential|secret|hidden)|"
    r"never\s+disclose|never\s+reveal)\s+([^.?!;\n]+)"
)

TYPE_DIRECTIVE_KEYWORDS: Dict[EntityType, List[str]] = {
    EntityType.EMAIL: ["email", "e-mail", "email address", "customer's email", "customer email", "support email", "support emails", "emails"],
    EntityType.PHONE: ["phone", "phone number", "mobile", "cell", "telephone", "contact number"],
    EntityType.BANK_ACCOUNT: ["account", "account number", "bank account", "account no", "iban", "customer account number", "customer's account"],
    EntityType.API_KEY: ["api key", "apikey", "api_key", "secret key"],
    EntityType.PASSWORD: ["password", "passcode", "secret", "credentials"],
    EntityType.ACCESS_TOKEN: ["token", "access token", "jwt", "bearer token"],
    EntityType.CREDIT_CARD: ["card", "credit card", "debit card", "card number"],
    EntityType.IP_ADDRESS: ["ip", "ip address", "server ip", "server ip address"],
    EntityType.ORDER_ID: ["order", "order id", "order number"],
    EntityType.TICKET_ID: ["ticket", "ticket id", "ticket number"],
    EntityType.CUSTOMER_ID: ["customer id", "customer's id", "customer number"],
    EntityType.USER_ID: ["user id", "account id"],
    EntityType.PERSON: ["name", "person", "customer name", "client name", "employee name"],
    EntityType.ORGANIZATION: ["company", "organization", "org", "firm", "business"],
    EntityType.LOCATION: ["location", "city", "place"],
}


def extract_linguistic_directives(prompt: str) -> Tuple[List[str], List[str]]:
    """Extract explicit inclusion and exclusion clauses from the prompt."""
    inclusions = [m.group(1).strip() for m in INCLUSION_DIRECTIVE_PATTERN.finditer(prompt)]
    exclusions = [m.group(1).strip() for m in EXCLUSION_DIRECTIVE_PATTERN.finditer(prompt)]
    return inclusions, exclusions


def check_entity_directives(
    entity: DetectedEntity,
    prompt: str,
    inclusions: List[str],
    exclusions: List[str],
) -> Tuple[Optional[str], bool]:
    """
    Check whether entity matches an explicit inclusion ("RETAIN") or exclusion ("MASK") directive.
    Returns (explicit_directive, disclosure_allowed).
    """
    ent_text_lower = entity.text.lower().strip()
    ent_type = entity.entity_type
    type_keywords = TYPE_DIRECTIVE_KEYWORDS.get(ent_type, [])

    # Check exclusions first (security policy takes precedence)
    for excl in exclusions:
        excl_lower = excl.lower()
        if ent_text_lower in excl_lower:
            # Check if this match is merely a possessive owner e.g. "John Smith's phone", "Sarah's email"
            possessive_match = re.search(rf"\b{re.escape(ent_text_lower)}'s\s+([a-z\s]+)", excl_lower)
            if possessive_match:
                # The entity is the owner/possessor, not the target secret being excluded
                pass
            else:
                return "MASK", False
        if any(re.search(rf"\b{re.escape(kw)}\b", excl_lower) for kw in type_keywords):
            return "MASK", False
        if "credential" in excl_lower and ent_type in (
            EntityType.PASSWORD, EntityType.API_KEY, EntityType.ACCESS_TOKEN, EntityType.RECOVERY_CODE
        ):
            return "MASK", False
        if "contact" in excl_lower and ent_type in (EntityType.EMAIL, EntityType.PHONE):
            return "MASK", False

    # Check inclusions
    for incl in inclusions:
        incl_lower = incl.lower()
        if ent_text_lower in incl_lower:
            return "RETAIN", True
        words = [w for w in re.split(r"\W+", ent_text_lower) if len(w) >= 3]
        if words and all(w in incl_lower for w in words):
            return "RETAIN", True
        if ent_type in (EntityType.ORDER_ID, EntityType.TICKET_ID, EntityType.CUSTOMER_ID):
            if any(re.search(rf"\b{re.escape(kw)}\b", incl_lower) for kw in type_keywords):
                return "RETAIN", True

    return None, True


class ContextAnalyzer:
    """
    Evaluates contextual roles of detected entities given the overall prompt and task type.
    Produces explainable ContextualRole objects for downstream policy assessment.
    """

    def _evaluate_value_requirement(
        self,
        entity: DetectedEntity,
        prompt: str,
        task_type: TaskType,
        operation: OperationType,
    ) -> Tuple[bool, EntityTaskRelation, str]:
        """
        Evaluate whether this entity's literal value is required for the requested task.
        """
        if re.search(r"(?i)\b(?:rewrite|rephrase|paraphrase|reword|proofread|sanitize|clean\s+up)\b", prompt):
            return False, EntityTaskRelation.TRANSFORMATION_CONTENT, "Text transformation content (literal value not required)"

        if re.search(r"(?i)\b(?:write|draft|compose)\s+(?:a\s+|an\s+)?(?:email|letter|application|cover\s+letter|memo|essay)\b", prompt):
            return False, EntityTaskRelation.OUTPUT_REFERENCE, "Output reference context (literal value not required)"

        if re.search(r"(?i)\b(?:explain|tell\s+me\s+about|what\s+is|who\s+is|describe)\s+(?:cloud\s+computing|ai|machine\s+learning|blockchain|quantum|the\s+difference|history|capital|python|java)\b", prompt):
            return False, EntityTaskRelation.IRRELEVANT, "Prompt targets external topic; entity is irrelevant"

        if operation == OperationType.COMPARISON:
            comp_match = re.search(r"(?i)\b(?:compare|comparison|difference\s+between)\s+([^.?!;\n]+)", prompt)
            if comp_match:
                comp_clause = comp_match.group(1).lower()
                if re.search(r"(?i)\b(?:java|python|c\+\+|golang|javascript|rust|react|angular|vue|aws|azure|gcp|ios|android)\s+(?:and|vs\.?|or)\s+(?:java|python|c\+\+|golang|javascript|rust|react|angular|vue|aws|azure|gcp|ios|android)\b", comp_clause):
                    return False, EntityTaskRelation.IRRELEVANT, "Comparison targets unrelated subjects; entity is irrelevant"

            is_comp_target = False
            for match in re.finditer(r"(?i)\b(?:compare|which(?:\s+[a-z]+)?\s+is\s+(?:stronger|better|longer|more\s+secure)|difference\s+between)\b", prompt):
                if abs(match.start() - entity.start) < 100 or abs(match.end() - entity.start) < 100:
                    is_comp_target = True
                    break

            if entity.entity_type == EntityType.PASSWORD and re.search(r"(?i)\bwhich\s+(?:password|one)\s+is\s+(?:stronger|better|longer|more\s+secure)\b", prompt):
                is_comp_target = True

            if is_comp_target:
                return True, EntityTaskRelation.COMPARISON_INPUT, "Entity is a direct comparison input requiring literal inspection"
            else:
                return False, EntityTaskRelation.IRRELEVANT, "Entity does not participate in the comparison"

        if operation == OperationType.VALUE_ANALYSIS:
            if entity.entity_type in (EntityType.PASSWORD, EntityType.API_KEY, EntityType.ACCESS_TOKEN, EntityType.RECOVERY_CODE):
                if re.search(
                    r"(?i)\b(?:is\s+(?:this|it|the|my)?\s*(?:password|key|token|string)?\s*(?:strong|weak|secure|safe|complex|compromised)|"
                    r"(?:password|key|token)\s+(?:strength|entropy|complexity|length)|"
                    r"how\s+many\s+(?:digits?|letters?|chars?|characters?|numbers?|symbols?)|"
                    r"count\s+(?:the\s+)?(?:digits?|letters?|chars?|characters?)|"
                    r"calculate\s+entropy)\b",
                    prompt,
                ):
                    return True, EntityTaskRelation.TARGET_OF_ANALYSIS, "Target of value analysis: password strength/complexity evaluation"

            if re.search(r"(?i)\b(?:how\s+many\s+(?:digits?|letters?|chars?|characters?)|count\s+(?:the\s+)?(?:digits?|letters?|chars?))\b", prompt):
                return True, EntityTaskRelation.TARGET_OF_ANALYSIS, "Target of lexical/character count analysis"

        if operation == OperationType.CALCULATION:
            if entity.entity_type in (EntityType.DOB, EntityType.DATE):
                if re.search(r"(?i)\b(?:age|how\s+old|years\s+old|difference\s+in\s+years|days\s+(?:between|until|since))\b", prompt):
                    return True, EntityTaskRelation.CALCULATION_INPUT, "Literal date value required for mathematical calculation of age/duration"

        if operation == OperationType.VALIDATION:
            if entity.entity_type == EntityType.IP_ADDRESS:
                if re.search(r"(?i)\b(?:private|public|routable|reserved|loopback|valid|ipv4|ipv6|subnet)\b", prompt):
                    return True, EntityTaskRelation.VALIDATION_INPUT, "Literal IP value required to validate network address class/range"
            elif entity.entity_type in (EntityType.PASSWORD, EntityType.API_KEY):
                if re.search(r"(?i)\b(?:valid|well-formed|meets?\s+(?:the\s+)?requirements|valid\s+password)\b", prompt):
                    return True, EntityTaskRelation.VALIDATION_INPUT, "Literal value required to validate policy/requirements"

        return False, EntityTaskRelation.NONE, "Literal value not required for task execution"

    def analyze_entity_context(
        self,
        entity: DetectedEntity,
        prompt: str,
        task_type: TaskType,
        operation: Optional[OperationType] = None,
    ) -> ContextualRole:
        """
        Evaluate single entity's context in the prompt.
        start/end offsets and normalized_value from DetectedEntity are threaded
        into every returned ContextualRole so Phase 4 can perform span-based masking.
        """
        if operation is None:
            operation = detect_operation(prompt)

        lower_text = entity.text.lower().strip()
        normalized_lower = entity.normalized_value.lower().strip()

        start_win = max(0, entity.start - 40)
        end_win = min(len(prompt), entity.end + 40)
        pre_context = prompt[start_win:entity.start]
        local_context = prompt[start_win:end_win]

        val_req, relation, rel_cue = self._evaluate_value_requirement(
            entity, prompt, task_type, operation
        )

        # Span shortcuts threaded into every ContextualRole for Phase 4
        _s = entity.start
        _e = entity.end
        _nv = entity.normalized_value

        # -----------------------------------------------------------------
        # 1. TECHNICAL CREDENTIALS
        # -----------------------------------------------------------------
        if entity.entity_type in (
            EntityType.API_KEY, EntityType.PASSWORD, EntityType.ACCESS_TOKEN,
            EntityType.RECOVERY_CODE, EntityType.CONNECTION_STRING,
        ):
            return ContextualRole(
                entity_text=entity.text,
                entity_type=entity.entity_type,
                role_category=RoleCategory.TECHNICAL_CREDENTIAL,
                is_first_party=True,
                is_public_knowledge=False,
                is_task_relevant=val_req,
                value_required=val_req,
                entity_task_relation=relation,
                context_cue=rel_cue if val_req else f"Strict technical credential of type {entity.entity_type.value}",
                confidence=0.99,
                start=_s, end=_e, normalized_value=_nv,
            )

        # -----------------------------------------------------------------
        # 2. PERSONAL IDENTIFIERS, GOVT IDs & HEALTHCARE PII
        # -----------------------------------------------------------------
        if entity.entity_type in (
            EntityType.NATIONAL_ID, EntityType.PASSPORT, EntityType.DRIVER_LICENSE,
            EntityType.TAX_ID, EntityType.CREDIT_CARD, EntityType.BANK_ACCOUNT,
            EntityType.MEDICAL_RECORD, EntityType.HEALTH_INSURANCE_ID,
            EntityType.MAC_ADDRESS, EntityType.IP_ADDRESS, EntityType.EMAIL,
            EntityType.PHONE, EntityType.USERNAME, EntityType.USER_ID,
            EntityType.CUSTOMER_ID, EntityType.ORDER_ID, EntityType.TICKET_ID,
            EntityType.DOB, EntityType.ADDRESS, EntityType.PII_OTHER,
        ):
            is_first_party = bool(FIRST_PARTY_CUES.search(local_context)) or "my" in pre_context.lower() or "i live" in pre_context.lower()
            is_task_rel = val_req or (task_type == TaskType.EMAIL_GENERATION and entity.entity_type in (EntityType.EMAIL, EntityType.PHONE))
            return ContextualRole(
                entity_text=entity.text,
                entity_type=entity.entity_type,
                role_category=RoleCategory.PERSONAL_IDENTIFIER,
                is_first_party=is_first_party,
                is_public_knowledge=False,
                is_task_relevant=is_task_rel,
                value_required=val_req,
                entity_task_relation=relation,
                context_cue=rel_cue if val_req else f"Personal sensitive identifier ({entity.entity_type.value}) detected",
                confidence=0.95,
                start=_s, end=_e, normalized_value=_nv,
            )

        # -----------------------------------------------------------------
        # 3. PERSON ENTITY EVALUATION
        # -----------------------------------------------------------------
        if entity.entity_type == EntityType.PERSON:
            if RECOVERY_QUESTION_CUES.search(local_context) or RECOVERY_QUESTION_CUES.search(pre_context):
                return ContextualRole(
                    entity_text=entity.text,
                    entity_type=EntityType.PERSON,
                    role_category=RoleCategory.TECHNICAL_CREDENTIAL,
                    is_first_party=True,
                    is_public_knowledge=False,
                    is_task_relevant=val_req,
                    value_required=val_req,
                    entity_task_relation=relation if val_req else EntityTaskRelation.NONE,
                    context_cue="Account recovery / security question sensitive answer (e.g. mother's maiden name)",
                    confidence=0.98,
                    start=_s, end=_e, normalized_value=_nv,
                )

            is_known_public = (
                lower_text in KNOWN_PUBLIC_FIGURES or normalized_lower in KNOWN_PUBLIC_FIGURES
            )

            is_general_reference = bool(
                re.search(r"(?i)\b(?:common\s+(?:surname|name|last\s+name|first\s+name)|meaning\s+of|origin\s+of)\b", local_context)
            )
            if is_general_reference and not FIRST_PARTY_CUES.search(pre_context):
                return ContextualRole(
                    entity_text=entity.text,
                    entity_type=EntityType.PERSON,
                    role_category=RoleCategory.GENERAL_REFERENCE,
                    is_first_party=False,
                    is_public_knowledge=True,
                    is_task_relevant=True,
                    context_cue="Common name mentioned in general/linguistic reference",
                    confidence=0.85,
                    start=_s, end=_e, normalized_value=_nv,
                )

            is_self_ident = bool(FIRST_PARTY_CUES.search(pre_context)) or bool(
                re.search(r"(?i)\b(?:my\s+name\s+is|i\s+am|call\s+me)\s+$", pre_context.strip())
            )
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
                    start=_s, end=_e, normalized_value=_nv,
                )

            return ContextualRole(
                entity_text=entity.text,
                entity_type=EntityType.PERSON,
                role_category=RoleCategory.PERSONAL_IDENTIFIER,
                is_first_party=is_self_ident,
                is_public_knowledge=False,
                is_task_relevant=(task_type == TaskType.EMAIL_GENERATION),
                context_cue="Personal name associated with user or individual",
                confidence=0.90,
                start=_s, end=_e, normalized_value=_nv,
            )

        # -----------------------------------------------------------------
        # 4. ORGANIZATION EVALUATION
        # -----------------------------------------------------------------
        if entity.entity_type == EntityType.ORGANIZATION:
            is_known_org = (
                lower_text in KNOWN_PUBLIC_ORGS or normalized_lower in KNOWN_PUBLIC_ORGS
            )
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
                    start=_s, end=_e, normalized_value=_nv,
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
                start=_s, end=_e, normalized_value=_nv,
            )

        # -----------------------------------------------------------------
        # 5. LOCATION EVALUATION
        # -----------------------------------------------------------------
        if entity.entity_type == EntityType.LOCATION:
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
                    start=_s, end=_e, normalized_value=_nv,
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
                start=_s, end=_e, normalized_value=_nv,
            )

        # -----------------------------------------------------------------
        # 6. DATE EVALUATION
        # -----------------------------------------------------------------
        if entity.entity_type == EntityType.DATE:
            is_dob = bool(re.search(r"(?i)\b(?:dob|date\s+of\s+birth|birth\s*date|birthday|born\s+(?:on)?)\b", local_context))
            if is_dob:
                is_first_party = bool(FIRST_PARTY_CUES.search(local_context)) or "my" in pre_context.lower()
                return ContextualRole(
                    entity_text=entity.text,
                    entity_type=EntityType.DOB,
                    role_category=RoleCategory.PERSONAL_IDENTIFIER,
                    is_first_party=is_first_party,
                    is_public_knowledge=False,
                    is_task_relevant=val_req,
                    value_required=val_req,
                    entity_task_relation=relation,
                    context_cue=rel_cue if val_req else "Personal date of birth / birthday identifier",
                    confidence=0.95,
                    start=_s, end=_e, normalized_value=_nv,
                )
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
                start=_s, end=_e, normalized_value=_nv,
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
            start=_s, end=_e, normalized_value=_nv,
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
        Extracts generalized linguistic privacy and inclusion directives and resolves coreferences.
        """
        operation = detect_operation(prompt)
        inclusions, exclusions = extract_linguistic_directives(prompt)

        contexts: List[ContextualRole] = []
        for ent in entities:
            role = self.analyze_entity_context(ent, prompt, task_type, operation=operation)
            
            # Apply explicit linguistic directives
            directive, disclosure = check_entity_directives(ent, prompt, inclusions, exclusions)
            if directive == "RETAIN":
                role.explicit_directive = "RETAIN"
                role.is_task_relevant = True
                role.disclosure_allowed = True
                role.role_category = RoleCategory.TASK_RELEVANT_ENTITY
                role.context_cue = f"Explicit directive to mention/retain entity ({role.entity_text}) in output"
            elif directive == "MASK":
                role.explicit_directive = "MASK"
                role.disclosure_allowed = False
                role.is_task_relevant = False
                role.context_cue = f"Explicit non-disclosure directive (do not expose) applying to {role.entity_text}"
            else:
                role.explicit_directive = None
                role.disclosure_allowed = disclosure

            contexts.append(role)

        # Propagate directives across entities with identical or contained normalized names (coreference resolution)
        for r in contexts:
            for other in contexts:
                if r is not other and r.entity_type == other.entity_type:
                    r_norm = r.normalized_value.lower()
                    o_norm = other.normalized_value.lower()
                    if r_norm == o_norm or (len(r_norm) >= 3 and (r_norm in o_norm or o_norm in r_norm)):
                        if other.explicit_directive == "RETAIN" and r.explicit_directive is None:
                            r.explicit_directive = "RETAIN"
                            r.is_task_relevant = True
                            r.role_category = other.role_category
                            r.disclosure_allowed = True
                        elif other.explicit_directive == "MASK" and r.explicit_directive is None:
                            r.explicit_directive = "MASK"
                            r.disclosure_allowed = False

        return ContextAnalysisResult(
            prompt=prompt,
            task_type=task_type,
            task_confidence=task_confidence,
            task_cues=task_cues,
            entity_contexts=contexts,
            operation_type=operation,
        )
