"""Core enums: capabilities, modalities, periods, frequencies, tier semantics."""

from enum import Enum


class Capability(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    AGENTIC = "agentic"  # tool use, multi-step planning
    MULTIMODAL = "multimodal"
    LONG_CONTEXT = "long_context"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CONTENT_WRITING = "content_writing"
    MATH = "math"
    OCR = "ocr"
    COMPLIANCE = "compliance"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


class Period(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class Frequency(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


TIER_SEMANTICS: dict[int, str] = {
    1: "basic",
    2: "limited",
    3: "competent",
    4: "strong",
    5: "top-of-class",
}

CAPABILITY_DESCRIPTIONS: dict[str, str] = {
    Capability.CODING.value: "Writing, reviewing, and debugging code.",
    Capability.REASONING.value: "Multi-step logical reasoning and analysis.",
    Capability.AGENTIC.value: "Tool use and multi-step planning/execution.",
    Capability.MULTIMODAL.value: "Understanding non-text inputs (images, audio, video).",
    Capability.LONG_CONTEXT.value: "Effective use of very large context windows.",
    Capability.INSTRUCTION_FOLLOWING.value: "Precise adherence to instructions and formats.",
    Capability.CONTENT_WRITING.value: "Long-form prose, marketing, and creative writing.",
    Capability.MATH.value: "Mathematical problem solving.",
    Capability.OCR.value: "Text extraction from documents and images.",
    Capability.COMPLIANCE.value: "Policy-sensitive and regulated-domain output quality.",
}
