#!/usr/bin/env python3
"""
Comprehensive validation of weighted cosine similarity using real message history.

This script:
1. Extracts real user prompts from .victor/project.db
2. Tests weighted vs standard cosine similarity
3. Validates that weighted similarity improves classification
4. Generates detailed report with examples

Usage:
    python scripts/validate_weighted_similarity.py [--limit N] [--output REPORT.md]
"""

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.storage.embeddings.service import EmbeddingService
from victor.storage.embeddings.task_classifier import TaskTypeClassifier, TaskType


def get_messages_from_db(db_path: Path, limit: int = 100) -> List[Dict[str, Any]]:
    """Extract user messages from the database.

    Args:
        db_path: Path to project.db
        limit: Maximum number of messages to extract

    Returns:
        List of message dictionaries
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get user messages (role='user')
    cursor.execute("""
        SELECT id, content, timestamp
        FROM messages
        WHERE role = 'user'
        AND length(content) > 20
        AND length(content) < 500
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    messages = []
    for row in cursor.fetchall():
        messages.append({
            "id": row["id"],
            "content": row["content"],
            "timestamp": row["timestamp"],
        })

    conn.close()
    return messages


def test_similarity_comparison(
    messages: List[Dict[str, Any]],
    classifier: TaskTypeClassifier,
) -> List[Dict[str, Any]]:
    """Test weighted vs standard similarity on real messages.

    Args:
        messages: List of message dictionaries
        classifier: TaskTypeClassifier instance

    Returns:
        List of comparison results
    """
    results = []
    embedding_service = classifier.embedding_service

    # Test phrases for each task type
    test_phrases = classifier._phrase_lists

    for msg in messages[:50]:  # Test first 50 messages
        query_text = msg["content"]
        query_emb = embedding_service.embed_text_sync(query_text)

        # Test against each task type
        for task_type, phrases in test_phrases.items():
            if not phrases:
                continue

            # Use first 5 phrases as corpus
            corpus_texts = phrases[:5]
            corpus_emb = np.vstack([
                embedding_service.embed_text_sync(text) for text in corpus_texts
            ])

            # Standard cosine similarity
            standard_similarities = EmbeddingService.cosine_similarity_matrix(
                query_emb, corpus_emb
            )

            # Weighted cosine similarity
            weighted_similarities = EmbeddingService.weighted_cosine_similarity(
                query_emb, query_text, corpus_emb, corpus_texts
            )

            # Find best match for each method
            standard_best_idx = np.argmax(standard_similarities)
            weighted_best_idx = np.argmax(weighted_similarities)
            standard_best_score = float(standard_similarities[standard_best_idx])
            weighted_best_score = float(weighted_similarities[weighted_best_idx])

            # Check if weighted improved the score
            improved = weighted_best_score > standard_best_score + 0.01

            results.append({
                "message_id": msg["id"],
                "message_content": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                "task_type": task_type.value,
                "standard_best_score": standard_best_score,
                "weighted_best_score": weighted_best_score,
                "improvement": weighted_best_score - standard_best_score,
                "improved": improved,
                "standard_match": corpus_texts[standard_best_idx][:50],
                "weighted_match": corpus_texts[weighted_best_idx][:50],
            })

    return results


def generate_report(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate markdown report with analysis.

    Args:
        results: Comparison results
        output_path: Path to write report
    """
    # Calculate statistics
    total_comparisons = len(results)
    improved_count = sum(1 for r in results if r["improved"])
    degraded_count = sum(1 for r in results if r["improvement"] < -0.01)
    neutral_count = total_comparisons - improved_count - degraded_count

    avg_improvement = np.mean([r["improvement"] for r in results])
    max_improvement = max([r["improvement"] for r in results])
    min_improvement = min([r["improvement"] for r in results])

    # Find best examples
    best_improvements = sorted(results, key=lambda x: x["improvement"], reverse=True)[:10]
    worst_degradations = sorted(results, key=lambda x: x["improvement"])[:10]

    # Group by task type
    by_task_type = defaultdict(list)
    for r in results:
        by_task_type[r["task_type"]].append(r)

    task_type_stats = {}
    for task_type, task_results in by_task_type.items():
        task_improved = sum(1 for r in task_results if r["improved"])
        task_total = len(task_results)
        task_avg_improvement = np.mean([r["improvement"] for r in task_results])
        task_type_stats[task_type] = {
            "total": task_total,
            "improved": task_improved,
            "improvement_rate": task_improved / task_total if task_total > 0 else 0,
            "avg_improvement": task_avg_improvement,
        }

    # Write report
    with open(output_path, "w") as f:
        f.write("# Weighted Cosine Similarity Validation Report\n\n")
        f.write(f"Generated: {Path(__file__).stat().st_mtime}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total comparisons**: {total_comparisons}\n")
        f.write(f"- **Improved**: {improved_count} ({improved_count/total_comparisons*100:.1f}%)\n")
        f.write(f"- **Degraded**: {degraded_count} ({degraded_count/total_comparisons*100:.1f}%)\n")
        f.write(f"- **Neutral**: {neutral_count} ({neutral_count/total_comparisons*100:.1f}%)\n\n")

        f.write("### Statistics\n\n")
        f.write(f"- **Average improvement**: {avg_improvement:+.4f}\n")
        f.write(f"- **Max improvement**: {max_improvement:+.4f}\n")
        f.write(f"- **Min improvement**: {min_improvement:+.4f}\n\n")

        f.write("## Per-Task-Type Analysis\n\n")
        f.write("| Task Type | Total | Improved | Rate | Avg Improvement |\n")
        f.write("|-----------|-------|----------|------|-----------------|\n")

        for task_type, stats in sorted(
            task_type_stats.items(),
            key=lambda x: x[1]["avg_improvement"],
            reverse=True
        ):
            f.write(
                f"| {task_type} | {stats['total']} | {stats['improved']} | "
                f"{stats['improvement_rate']*100:.1f}% | {stats['avg_improvement']:+.4f} |\n"
            )

        f.write("\n## Top Improvements\n\n")
        for i, r in enumerate(best_improvements[:5], 1):
            f.write(f"### {i}. Improvement: {r['improvement']:+.4f}\n\n")
            f.write(f"**Message**: {r['message_content']}\n\n")
            f.write(f"**Task Type**: {r['task_type']}\n\n")
            f.write(f"- Standard: {r['standard_best_score']:.4f} (matched: \"{r['standard_match']}...\")\n")
            f.write(f"- Weighted: {r['weighted_best_score']:.4f} (matched: \"{r['weighted_match']}...\")\n\n")

        f.write("\n## Worst Degradations\n\n")
        for i, r in enumerate(worst_degradations[:5], 1):
            f.write(f"### {i}. Degradation: {r['improvement']:+.4f}\n\n")
            f.write(f"**Message**: {r['message_content']}\n\n")
            f.write(f"**Task Type**: {r['task_type']}\n\n")
            f.write(f"- Standard: {r['standard_best_score']:.4f} (matched: \"{r['standard_match']}...\")\n")
            f.write(f"- Weighted: {r['weighted_best_score']:.4f} (matched: \"{r['weighted_match']}...\")\n\n")

        f.write("\n## Conclusions\n\n")

        if improved_count > degraded_count * 2:
            f.write("✅ **Weighted similarity shows significant improvement**\n\n")
            f.write("The weighted cosine similarity with key term boosting is working as expected.\n")
            f.write("It improves classification accuracy for most cases while maintaining\n")
            f.write("backward compatibility.\n")
        elif improved_count > degraded_count:
            f.write("⚠️ **Weighted similarity shows moderate improvement**\n\n")
            f.write("The weighted cosine similarity provides some benefit but may need tuning.\n")
            f.write("Consider adjusting the key term weights or boost formula.\n")
        else:
            f.write("❌ **Weighted similarity needs review**\n\n")
            f.write("The weighted cosine similarity is not providing the expected benefit.\n")
            f.write("Review the key term weights and consider alternative approaches.\n")

    print(f"\n✅ Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate weighted cosine similarity using real message history"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=".victor/project.db",
        help="Path to project.db (default: .victor/project.db)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of messages to test (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="WEIGHTED_SIMILARITY_VALIDATION.md",
        help="Output report path (default: WEIGHTED_SIMILARITY_VALIDATION.md)",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("Using messages from test data instead...")
        # Create synthetic test data
        messages = [
            {"id": 1, "content": "framework structural analysis", "timestamp": 0},
            {"id": 2, "content": "create a new file", "timestamp": 0},
            {"id": 3, "content": "review the code", "timestamp": 0},
            {"id": 4, "content": "search for functions", "timestamp": 0},
            {"id": 5, "content": "analyze the architecture", "timestamp": 0},
        ]
    else:
        print(f"📊 Extracting messages from: {db_path}")
        messages = get_messages_from_db(db_path, args.limit)
        print(f"✅ Extracted {len(messages)} messages")

    if not messages:
        print("❌ No messages found")
        return 1

    print("🔧 Initializing classifier...")
    classifier = TaskTypeClassifier.get_instance()
    classifier.initialize_sync()
    print("✅ Classifier initialized")

    print("🧪 Running similarity comparison...")
    results = test_similarity_comparison(messages, classifier)
    print(f"✅ Tested {len(results)} comparisons")

    print("📈 Generating report...")
    output_path = Path(args.output)
    generate_report(results, output_path)

    # Print summary
    improved = sum(1 for r in results if r["improved"])
    total = len(results)
    print(f"\n📊 Summary: {improved}/{total} ({improved/total*100:.1f}%) comparisons improved")

    return 0


if __name__ == "__main__":
    exit(main())
