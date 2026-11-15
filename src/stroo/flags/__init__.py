"""Detection of informal flags: vocabulary, patterns, matcher and filters."""

from .matcher import LABELS, FlagMatcher, get_matcher
from .patterns import PATTERNS
from .vocabulary import FLAG_TYPES, deps, pos, voc

__all__ = ["FLAG_TYPES", "LABELS", "PATTERNS", "FlagMatcher", "deps", "get_matcher", "pos", "voc"]
