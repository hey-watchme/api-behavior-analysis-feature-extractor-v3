"""
Event Filter for PaSST Behavior Detection
Filter out false positive events based on label-specific thresholds
"""

import json
import os
from typing import List, Dict, Optional


class EventFilter:
    """
    Filter predicted events based on configuration rules

    Filtering Strategy:
    1. Apply global threshold (remove events below minimum score)
    2. Apply label-specific thresholds (remove likely false positives)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize event filter with configuration

        Args:
            config_path: Path to filter_config.json (default: same directory)
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'filter_config.json')

        self.config = self._load_config(config_path)
        self.global_threshold = self.config.get('global_threshold', 0.1)
        self.label_thresholds = self.config.get('label_specific_thresholds', {})

        print(f"Event filter initialized:")
        print(f"  - Global threshold: {self.global_threshold}")
        print(f"  - Label-specific rules: {len(self.label_thresholds)} labels")

    def _load_config(self, config_path: str) -> Dict:
        """Load filter configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Filter config loaded: {config_path}")
            return config
        except FileNotFoundError:
            print(f"Warning: Filter config not found at {config_path}, using defaults")
            return {'global_threshold': 0.1, 'label_specific_thresholds': {}}
        except Exception as e:
            print(f"Error loading filter config: {e}")
            return {'global_threshold': 0.1, 'label_specific_thresholds': {}}

    def filter_events(self, events: List[Dict]) -> List[Dict]:
        """
        Filter events based on configuration rules

        Args:
            events: List of predicted events with 'label' and 'score' fields

        Returns:
            Filtered list of events
        """
        if not events:
            return []

        filtered = []

        for event in events:
            label = event.get('label', '')
            score = event.get('score', 0.0)

            # Step 1: Apply global threshold
            if score < self.global_threshold:
                continue

            # Step 2: Apply label-specific threshold
            if label in self.label_thresholds:
                label_threshold = self.label_thresholds[label]
                if score < label_threshold:
                    # Event filtered out due to label-specific threshold
                    continue

            # Event passed all filters
            filtered.append(event)

        return filtered

    def filter_timeline(self, timeline: List[Dict]) -> List[Dict]:
        """
        Filter timeline data (list of time blocks with events)

        Args:
            timeline: List of time blocks, each containing 'time' and 'events'

        Returns:
            Timeline with filtered events
        """
        filtered_timeline = []

        for time_block in timeline:
            time_value = time_block.get('time', 0.0)
            events = time_block.get('events', [])

            # Filter events in this time block
            filtered_events = self.filter_events(events)

            # Keep time block even if all events are filtered (preserve timeline structure)
            filtered_timeline.append({
                'time': time_value,
                'events': filtered_events
            })

        return filtered_timeline

    def get_filter_stats(self, original_events: List[Dict], filtered_events: List[Dict]) -> Dict:
        """
        Calculate filtering statistics

        Args:
            original_events: Events before filtering
            filtered_events: Events after filtering

        Returns:
            Statistics dictionary
        """
        original_count = len(original_events)
        filtered_count = len(filtered_events)
        removed_count = original_count - filtered_count

        return {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': removed_count,
            'removal_rate': round(removed_count / original_count, 3) if original_count > 0 else 0.0
        }


# Global filter instance (initialized once)
_filter_instance = None

def get_event_filter() -> EventFilter:
    """Get or create global event filter instance"""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = EventFilter()
    return _filter_instance
