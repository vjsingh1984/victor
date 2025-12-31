// Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! High-performance multi-pattern matching using Aho-Corasick.
//!
//! This module provides 10-100x faster multi-pattern matching compared to
//! Python re module for keyword detection, tool selection, and task classification.
//!
//! Use cases:
//! - Tool keyword matching (find all matching tools for a query)
//! - Intent detection (match multiple intent patterns simultaneously)
//! - Task classification (detect action/analysis/generation keywords)
//! - Entity extraction (find all entity mentions in text)

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// A match result from pattern matching
#[pyclass]
#[derive(Clone)]
pub struct PatternMatch {
    /// The pattern index that matched
    #[pyo3(get)]
    pub pattern_idx: usize,
    /// The matched text
    #[pyo3(get)]
    pub matched_text: String,
    /// Start position in text
    #[pyo3(get)]
    pub start: usize,
    /// End position in text
    #[pyo3(get)]
    pub end: usize,
}

#[pymethods]
impl PatternMatch {
    fn __repr__(&self) -> String {
        format!(
            "PatternMatch(pattern={}, text='{}', span=({}, {}))",
            self.pattern_idx, self.matched_text, self.start, self.end
        )
    }
}

/// Fast multi-pattern matcher using Aho-Corasick algorithm
#[pyclass]
pub struct PatternMatcher {
    /// The compiled Aho-Corasick automaton
    automaton: Arc<AhoCorasick>,
    /// Original patterns for reference
    patterns: Vec<String>,
    /// Whether matching is case-insensitive
    #[pyo3(get)]
    case_insensitive: bool,
}

#[pymethods]
impl PatternMatcher {
    /// Create a new pattern matcher from a list of patterns
    #[new]
    #[pyo3(signature = (patterns, case_insensitive=true))]
    pub fn new(patterns: Vec<String>, case_insensitive: bool) -> PyResult<Self> {
        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(case_insensitive)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            automaton: Arc::new(automaton),
            patterns,
            case_insensitive,
        })
    }

    /// Find all matches in text
    pub fn find_all(&self, text: &str) -> Vec<PatternMatch> {
        self.automaton
            .find_iter(text)
            .map(|m| PatternMatch {
                pattern_idx: m.pattern().as_usize(),
                matched_text: text[m.start()..m.end()].to_string(),
                start: m.start(),
                end: m.end(),
            })
            .collect()
    }

    /// Check if text contains any pattern
    pub fn contains_any(&self, text: &str) -> bool {
        self.automaton.is_match(text)
    }

    /// Count total number of matches
    pub fn count_matches(&self, text: &str) -> usize {
        self.automaton.find_iter(text).count()
    }

    /// Get unique pattern indices that matched
    pub fn matched_patterns(&self, text: &str) -> Vec<usize> {
        let mut seen = vec![false; self.patterns.len()];
        let mut result = Vec::new();

        for m in self.automaton.find_iter(text) {
            let idx = m.pattern().as_usize();
            if !seen[idx] {
                seen[idx] = true;
                result.push(idx);
            }
        }

        result
    }

    /// Get pattern strings that matched
    pub fn matched_pattern_strings(&self, text: &str) -> Vec<String> {
        self.matched_patterns(text)
            .into_iter()
            .map(|i| self.patterns[i].clone())
            .collect()
    }

    /// Count matches per pattern
    pub fn count_by_pattern(&self, text: &str) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for m in self.automaton.find_iter(text) {
            *counts.entry(m.pattern().as_usize()).or_insert(0) += 1;
        }
        counts
    }

    /// Get pattern at index
    pub fn get_pattern(&self, idx: usize) -> Option<String> {
        self.patterns.get(idx).cloned()
    }

    /// Get number of patterns
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Replace all matches with a replacement string
    #[pyo3(signature = (text, replacement))]
    pub fn replace_all(&self, text: &str, replacement: &str) -> String {
        // Create a replacement for each pattern
        let replacements: Vec<&str> = vec![replacement; self.patterns.len()];
        self.automaton.replace_all(text, &replacements)
    }

    /// Replace matches with pattern-specific replacements
    #[pyo3(signature = (text, replacements))]
    pub fn replace_all_with(&self, text: &str, replacements: Vec<String>) -> PyResult<String> {
        if replacements.len() != self.patterns.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} replacements, got {}",
                self.patterns.len(),
                replacements.len()
            )));
        }
        let refs: Vec<&str> = replacements.iter().map(|s| s.as_str()).collect();
        Ok(self.automaton.replace_all(text, &refs))
    }
}

/// Quick function to check if text contains any of the patterns
#[pyfunction]
#[pyo3(signature = (text, patterns, case_insensitive=true))]
pub fn contains_any_pattern(text: &str, patterns: Vec<String>, case_insensitive: bool) -> bool {
    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(case_insensitive)
        .build(&patterns);

    match ac {
        Ok(ac) => ac.is_match(text),
        Err(_) => false,
    }
}

/// Quick function to find all pattern matches in text
#[pyfunction]
#[pyo3(signature = (text, patterns, case_insensitive=true))]
pub fn find_all_patterns(
    text: &str,
    patterns: Vec<String>,
    case_insensitive: bool,
) -> Vec<PatternMatch> {
    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(case_insensitive)
        .match_kind(MatchKind::LeftmostLongest)
        .build(&patterns);

    match ac {
        Ok(ac) => ac
            .find_iter(text)
            .map(|m| PatternMatch {
                pattern_idx: m.pattern().as_usize(),
                matched_text: text[m.start()..m.end()].to_string(),
                start: m.start(),
                end: m.end(),
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Quick function to count pattern matches
#[pyfunction]
#[pyo3(signature = (text, patterns, case_insensitive=true))]
pub fn count_pattern_matches(text: &str, patterns: Vec<String>, case_insensitive: bool) -> usize {
    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(case_insensitive)
        .build(&patterns);

    match ac {
        Ok(ac) => ac.find_iter(text).count(),
        Err(_) => 0,
    }
}

/// Quick function to get indices of matched patterns
#[pyfunction]
#[pyo3(signature = (text, patterns, case_insensitive=true))]
pub fn get_matched_pattern_indices(
    text: &str,
    patterns: Vec<String>,
    case_insensitive: bool,
) -> Vec<usize> {
    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(case_insensitive)
        .build(&patterns);

    match ac {
        Ok(ac) => {
            let mut seen = vec![false; patterns.len()];
            let mut result = Vec::new();

            for m in ac.find_iter(text) {
                let idx = m.pattern().as_usize();
                if !seen[idx] {
                    seen[idx] = true;
                    result.push(idx);
                }
            }
            result
        }
        Err(_) => Vec::new(),
    }
}

/// Batch check multiple texts against the same patterns
#[pyfunction]
#[pyo3(signature = (texts, patterns, case_insensitive=true))]
pub fn batch_contains_any(
    texts: Vec<String>,
    patterns: Vec<String>,
    case_insensitive: bool,
) -> Vec<bool> {
    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(case_insensitive)
        .build(&patterns);

    match ac {
        Ok(ac) => texts.iter().map(|t| ac.is_match(t)).collect(),
        Err(_) => vec![false; texts.len()],
    }
}

/// Calculate a weighted score for pattern matches
/// Each pattern can have a weight, and the function returns the sum of weights
/// for all matched patterns (unique patterns only).
#[pyfunction]
#[pyo3(signature = (text, patterns, weights, case_insensitive=true))]
pub fn weighted_pattern_score(
    text: &str,
    patterns: Vec<String>,
    weights: Vec<f64>,
    case_insensitive: bool,
) -> PyResult<f64> {
    if patterns.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Number of patterns ({}) must match number of weights ({})",
            patterns.len(),
            weights.len()
        )));
    }

    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(case_insensitive)
        .build(&patterns)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let mut seen = vec![false; patterns.len()];
    let mut score = 0.0;

    for m in ac.find_iter(text) {
        let idx = m.pattern().as_usize();
        if !seen[idx] {
            seen[idx] = true;
            score += weights[idx];
        }
    }

    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matcher_creation() {
        let patterns = vec!["hello".to_string(), "world".to_string()];
        let matcher = PatternMatcher::new(patterns, true).unwrap();
        assert_eq!(matcher.pattern_count(), 2);
    }

    #[test]
    fn test_find_all() {
        let patterns = vec!["hello".to_string(), "world".to_string()];
        let matcher = PatternMatcher::new(patterns, true).unwrap();
        let matches = matcher.find_all("Hello World Hello");
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_contains_any() {
        let patterns = vec!["hello".to_string(), "world".to_string()];
        let matcher = PatternMatcher::new(patterns, true).unwrap();
        assert!(matcher.contains_any("Hello there"));
        assert!(!matcher.contains_any("Goodbye"));
    }

    #[test]
    fn test_case_insensitive() {
        let patterns = vec!["HELLO".to_string()];
        let matcher = PatternMatcher::new(patterns, true).unwrap();
        assert!(matcher.contains_any("hello"));

        let patterns2 = vec!["HELLO".to_string()];
        let matcher2 = PatternMatcher::new(patterns2, false).unwrap();
        assert!(!matcher2.contains_any("hello"));
    }

    #[test]
    fn test_weighted_score() {
        let patterns = vec![
            "read".to_string(),
            "write".to_string(),
            "analyze".to_string(),
        ];
        let weights = vec![1.0, 2.0, 3.0];
        let score = weighted_pattern_score("Please read and analyze this", patterns, weights, true)
            .unwrap();
        assert_eq!(score, 4.0); // read (1.0) + analyze (3.0)
    }
}
