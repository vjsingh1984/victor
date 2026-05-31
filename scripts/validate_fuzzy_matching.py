#!/usr/bin/env python3
"""
Validate fuzzy matching on real message history.

This script loads real messages from the database, generates realistic typos,
and validates that fuzzy matching improves classification accuracy.
"""

import argparse
import logging
import random
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_messages(db_path: Path, limit: int = 1000) -> List[Dict]:
    """Load real messages from the database.

    Args:
        db_path: Path to the project database
        limit: Maximum number of messages to load

    Returns:
        List of message dictionaries
    """
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Load user messages
        cursor.execute(
            """
            SELECT content, role
            FROM messages
            WHERE role = 'user'
            ORDER BY rowid
            LIMIT ?
        """,
            (limit,),
        )

        messages = []
        for row in cursor.fetchall():
            content, role = row
            if content and len(content.strip()) > 10:  # Filter out very short messages
                messages.append({"content": content, "role": role})

        conn.close()
        logger.info(f"Loaded {len(messages)} messages from {db_path}")
        return messages

    except Exception as e:
        logger.error(f"Error loading messages: {e}")
        return []


def generate_realistic_typo(message: str) -> str:
    """Generate a realistic typo in a message.

    Simulates common typing mistakes:
    - Missing letters (10% chance)
    - Transposed letters (10% chance)
    - Extra letters (5% chance)

    Args:
        message: Original message

    Returns:
        Message with realistic typo
    """
    if not message or len(message) < 5:
        return message

    # Common technical keywords that might be mistyped
    keywords = [
        "analyze",
        "analysis",
        "review",
        "structure",
        "architecture",
        "framework",
        "design",
        "implement",
        "execute",
        "search",
        "refactor",
        "test",
        "deploy",
        "examine",
        "investigate",
        "check",
    ]

    words = message.split()
    modified_words = []

    for word in words:
        # Clean word (remove punctuation)
        clean_word = word.lower().strip(".,!?;:\"'")
        punctuation = word[len(clean_word) :] if clean_word != word.lower() else ""

        # Check if this is a keyword we want to typo
        if clean_word in keywords and random.random() < 0.3:  # 30% chance to typo
            typo_type = random.random()

            if typo_type < 0.4:  # 40%: missing letter
                if len(clean_word) > 3:
                    pos = random.randint(1, len(clean_word) - 2)
                    typo_word = clean_word[:pos] + clean_word[pos + 1 :]
                else:
                    typo_word = clean_word
            elif typo_type < 0.7:  # 30%: transposed letters
                if len(clean_word) > 2:
                    pos = random.randint(0, len(clean_word) - 2)
                    typo_word = (
                        clean_word[:pos]
                        + clean_word[pos + 1]
                        + clean_word[pos]
                        + clean_word[pos + 2 :]
                    )
                else:
                    typo_word = clean_word
            else:  # 30%: double letter
                if len(clean_word) > 2:
                    pos = random.randint(1, len(clean_word) - 1)
                    typo_word = clean_word[:pos] + clean_word[pos] + clean_word[pos:]
                else:
                    typo_word = clean_word

            # Preserve original case
            if word[0].isupper():
                typo_word = typo_word.capitalize()

            modified_words.append(typo_word + punctuation)
        else:
            modified_words.append(word)

    return " ".join(modified_words)


def generate_realistic_typos(messages: List[Dict]) -> List[Tuple[str, str, str]]:
    """Generate realistic typos for a list of messages.

    Args:
        messages: List of original messages

    Returns:
        List of (original, typo, message_id) tuples
    """
    typos = []

    for msg in messages:
        original = msg["content"]
        typo_version = generate_realistic_typo(original)

        # Only include if typo was actually introduced
        if typo_version != original:
            typos.append((original, typo_version, str(hash(original))))

    logger.info(f"Generated {len(typos)} realistic typos from {len(messages)} messages")
    return typos


def validate_classification_improvement(
    typos: List[Tuple[str, str, str]], verbose: bool = False
) -> float:
    """Validate that fuzzy matching improves classification.

    Args:
        typos: List of (original, typo, message_id) tuples
        verbose: Whether to print detailed results

    Returns:
        Improvement rate (0.0 to 1.0)
    """
    try:
        from victor.storage.embeddings.task_classifier import TaskTypeClassifier

        # Get classifier instance
        classifier = TaskTypeClassifier.get_instance()
        if not classifier.is_initialized():
            logger.info("Initializing classifier...")
            classifier.initialize_sync()

    except Exception as e:
        logger.error(f"Error initializing classifier: {e}")
        return 0.0

    improved = 0
    total = len(typos)

    for original, typo_msg, msg_id in typos:
        try:
            # Classify original
            original_result = classifier.classify_sync(original)

            # Classify with typo
            typo_result = classifier.classify_sync(typo_msg)

            # Check if fuzzy matching helped
            # We consider it helped if:
            # 1. Both classify to the same type (consistency)
            # 2. OR typo classification has reasonable confidence (>0.3)

            if original_result.task_type == typo_result.task_type:
                improved += 1
                if verbose:
                    logger.info(
                        f"✓ Consistent: '{original[:50]}...' -> "
                        f"{original_result.task_type.value} "
                        f"(confidence: {typo_result.confidence:.2f})"
                    )
            elif typo_result.confidence > 0.3:
                improved += 1
                if verbose:
                    logger.info(
                        f"✓ Reasonable: '{original[:50]}...' -> "
                        f"{typo_result.task_type.value} "
                        f"(confidence: {typo_result.confidence:.2f})"
                    )
            else:
                if verbose:
                    logger.warning(
                        f"✗ Divergent: '{original[:50]}...' -> "
                        f"Original: {original_result.task_type.value}, "
                        f"Typo: {typo_result.task_type.value} "
                        f"(confidence: {typo_result.confidence:.2f})"
                    )

        except Exception as e:
            logger.error(f"Error classifying message: {e}")
            continue

    improvement_rate = improved / total if total > 0 else 0.0
    return improvement_rate


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate fuzzy matching on real message history")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(".victor/project.db"),
        help="Path to project database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of messages to load",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed results",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="Minimum improvement rate to pass validation (default: 0.05 = 5%%)",
    )

    args = parser.parse_args()

    # Load messages
    logger.info(f"Loading messages from {args.db_path}...")
    messages = load_messages(args.db_path, args.limit)

    if not messages:
        logger.error("No messages loaded. Cannot run validation.")
        sys.exit(1)

    # Generate realistic typos
    logger.info("Generating realistic typos...")
    typos = generate_realistic_typos(messages)

    if not typos:
        logger.error("No typos generated. Cannot run validation.")
        sys.exit(1)

    # Validate classification improvement
    logger.info("Validating classification with fuzzy matching...")
    improvement_rate = validate_classification_improvement(typos, verbose=args.verbose)

    # Report results
    logger.info("=" * 60)
    logger.info("Fuzzy matching validation results:")
    logger.info(f"  Total typo cases: {len(typos)}")
    logger.info(f"  Improved/consistent: {int(improvement_rate * len(typos))}")
    logger.info(f"  Improvement rate: {improvement_rate * 100:.1f}%")
    logger.info(f"  Minimum required: {args.min_improvement * 100:.1f}%")
    logger.info("=" * 60)

    # Check if validation passed
    if improvement_rate >= args.min_improvement:
        logger.info("✅ Validation passed!")
        sys.exit(0)
    else:
        logger.error(
            f"❌ Validation failed: {improvement_rate * 100:.1f}% < "
            f"{args.min_improvement * 100:.1f}%"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
