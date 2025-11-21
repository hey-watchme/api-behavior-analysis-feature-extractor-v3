#!/usr/bin/env python3
"""
Test script for event filtering
Verify that filter_config.json rules are applied correctly
"""

from event_filter import EventFilter

def test_filtering():
    """Test event filtering with sample data"""

    # Initialize filter
    event_filter = EventFilter()

    # Sample events (simulated predictions)
    sample_events = [
        {"label": "Speech / 会話・発話", "score": 0.85},           # Should pass (high score)
        {"label": "Dog / 犬", "score": 0.25},                      # Should be filtered (below 0.3)
        {"label": "Dog / 犬", "score": 0.40},                      # Should pass (above 0.3)
        {"label": "Laughter / 笑い声", "score": 0.60},             # Should pass
        {"label": "Bird vocalization / 鳥の鳴き声", "score": 0.15}, # Should be filtered (below 0.3)
        {"label": "Machine gun / 機関銃", "score": 0.28},          # Should be filtered (below 0.3)
        {"label": "Music / 音楽", "score": 0.50},                  # Should pass (no specific threshold)
        {"label": "Gunshot / 銃声", "score": 0.35},                # Should pass (above 0.3)
        {"label": "Silence / 沈黙", "score": 0.05},                # Should be filtered (below global 0.1)
    ]

    print("=" * 60)
    print("Event Filtering Test")
    print("=" * 60)
    print(f"\nFilter Configuration:")
    print(f"  Global threshold: {event_filter.global_threshold}")
    print(f"  Label-specific rules: {len(event_filter.label_thresholds)} labels")
    print()

    print(f"Original events: {len(sample_events)}")
    print("-" * 60)
    for i, event in enumerate(sample_events, 1):
        label = event['label']
        score = event['score']
        threshold = event_filter.label_thresholds.get(label, event_filter.global_threshold)
        will_pass = score >= threshold
        status = "✅ PASS" if will_pass else "❌ FILTERED"
        print(f"{i}. {status} | {label[:40]:40s} | score={score:.2f} (threshold={threshold:.2f})")

    print()

    # Apply filter
    filtered_events = event_filter.filter_events(sample_events)

    # Get statistics
    stats = event_filter.get_filter_stats(sample_events, filtered_events)

    print("=" * 60)
    print("Filtering Results")
    print("=" * 60)
    print(f"  Original count: {stats['original_count']}")
    print(f"  Filtered count: {stats['filtered_count']}")
    print(f"  Removed count:  {stats['removed_count']}")
    print(f"  Removal rate:   {stats['removal_rate'] * 100:.1f}%")
    print()

    print("Events after filtering:")
    print("-" * 60)
    for i, event in enumerate(filtered_events, 1):
        print(f"{i}. {event['label'][:40]:40s} | score={event['score']:.2f}")

    print()
    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

def test_timeline_filtering():
    """Test timeline filtering"""

    event_filter = EventFilter()

    # Sample timeline data (10-second blocks)
    sample_timeline = [
        {
            "time": 0.0,
            "events": [
                {"label": "Speech / 会話・発話", "score": 0.75},
                {"label": "Dog / 犬", "score": 0.20},  # Will be filtered
            ]
        },
        {
            "time": 10.0,
            "events": [
                {"label": "Laughter / 笑い声", "score": 0.60},
                {"label": "Bird vocalization / 鳥の鳴き声", "score": 0.15},  # Will be filtered
            ]
        },
        {
            "time": 20.0,
            "events": [
                {"label": "Machine gun / 機関銃", "score": 0.25},  # Will be filtered
            ]
        }
    ]

    print("\n" + "=" * 60)
    print("Timeline Filtering Test")
    print("=" * 60)

    print(f"\nOriginal timeline: {len(sample_timeline)} time blocks")
    for block in sample_timeline:
        print(f"  {block['time']}s: {len(block['events'])} events")

    # Apply timeline filtering
    filtered_timeline = event_filter.filter_timeline(sample_timeline)

    print(f"\nFiltered timeline: {len(filtered_timeline)} time blocks")
    for block in filtered_timeline:
        print(f"  {block['time']}s: {len(block['events'])} events (after filtering)")
        for event in block['events']:
            print(f"    - {event['label'][:40]:40s} | score={event['score']:.2f}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_filtering()
    test_timeline_filtering()
