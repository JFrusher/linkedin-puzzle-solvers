"""Functional action-play checks for Pinpoint using ConceptNet and collocation validation."""

from __future__ import annotations

import re
import requests
from collections import defaultdict


def _clean_clue(clue: str) -> str:
    """Strip determiners and formatting for concept queries."""
    clean = clue.strip().lower()
    clean = re.sub(r"^(the|a|an)\s+", "", clean)
    return clean


def get_conceptnet_actions(clue: str) -> set[str]:
    """Query ConceptNet for verbs/actions associated with a noun clue via UsedFor or CapableOf."""
    clean_clue = _clean_clue(clue)
    url = f"https://api.conceptnet.io/query?node=/c/en/{clean_clue}&rel=/r/UsedFor&limit=20"
    
    actions = set()
    try:
        response = requests.get(url, timeout=3).json()
        for edge in response.get("edges", []):
            end_term = edge.get("End", {}).get("label", "")
            # Keep single-word verbs or simple verb phrases
            words = end_term.lower().split()
            if 1 <= len(words) <= 2 and words[0].isalpha():
                actions.add(words[0])
    except Exception:
        pass
        
    # Also check CapableOfReceivingAction
    url_receives = f"https://api.conceptnet.io/query?node=/c/en/{clean_clue}&rel=/r/CapableOfReceivingAction&limit=20"
    try:
        response = requests.get(url_receives, timeout=3).json()
        for edge in response.get("edges", []):
            end_term = edge.get("End", {}).get("label", "")
            words = end_term.lower().split()
            if 1 <= len(words) <= 2 and words[0].isalpha():
                actions.add(words[0])
    except Exception:
        pass
        
    return actions


def score_actions(clues: list[str]) -> dict[str, int]:
    """Gather action candidates from all clues and score them by intersection coverage."""
    if not clues:
        return {}
        
    candidate_counts: defaultdict[str, int] = defaultdict(int)
    clue_actions_list = [get_conceptnet_actions(c) for c in clues]
    
    # Count how many clues support each action
    all_actions = set().union(*clue_actions_list) if clue_actions_list else set()
    for action in all_actions:
        hits = sum(1 for actions in clue_actions_list if action in actions)
        if hits >= 2:  # Must apply to at least 2 clues to be a viable category
            candidate_counts[f"things you can {action}"] = hits
            
    return dict(candidate_counts)