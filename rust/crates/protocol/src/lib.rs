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

//! Victor Protocol Types
//!
//! Portable message and protocol types for the Victor AI framework.
//! These types are designed for use in edge deployments and any context
//! where PyO3 is not available. They mirror the Python Pydantic models
//! used in `victor/providers/` and `victor/framework/events.py`.
//!
//! # Design
//!
//! - Pure Rust with serde (no PyO3 dependency)
//! - All types derive `Serialize`, `Deserialize`, `Clone`, `Debug`
//! - Optional fields use `skip_serializing_if` to produce clean JSON
//!
//! # Example
//!
//! ```rust
//! use victor_protocol::{Message, ToolCall, CompletionResponse, Usage};
//!
//! let msg = Message {
//!     role: "user".to_string(),
//!     content: Some("Hello, world!".to_string()),
//!     tool_calls: None,
//!     tool_call_id: None,
//! };
//!
//! let json = serde_json::to_string(&msg).unwrap();
//! assert!(json.contains("\"role\":\"user\""));
//! ```

use serde::{Deserialize, Serialize};

/// A chat message in the LLM conversation protocol.
///
/// Mirrors `victor.providers.base.Message` in Python.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Message {
    /// The role of the message sender (e.g., "system", "user", "assistant", "tool").
    pub role: String,

    /// The text content of the message. May be `None` for tool-call-only assistant messages.
    pub content: Option<String>,

    /// Tool calls requested by the assistant. Present only in assistant messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// The ID of the tool call this message is responding to. Present only in tool messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool invocation requested by the model.
///
/// Mirrors the tool_call structure used across all Victor providers.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,

    /// Name of the tool to invoke.
    pub name: String,

    /// Arguments to pass to the tool, as a JSON value.
    pub arguments: serde_json::Value,
}

/// A completion response from an LLM provider.
///
/// Mirrors `victor.providers.base.CompletionResponse` in Python.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CompletionResponse {
    /// The text content of the response.
    pub content: Option<String>,

    /// Tool calls requested by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// The model identifier that generated this response.
    pub model: String,

    /// The reason the model stopped generating (e.g., "stop", "tool_use", "length").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

/// Token usage statistics for a completion request.
///
/// Mirrors `victor.providers.base.Usage` in Python.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,

    /// Number of tokens in the completion.
    pub completion_tokens: usize,

    /// Total tokens used (prompt + completion).
    pub total_tokens: usize,
}

/// A tool definition for the LLM to use.
///
/// Mirrors `victor.framework.tools.ToolDefinition` in Python.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolDefinition {
    /// The name of the tool.
    pub name: String,

    /// A description of what the tool does.
    pub description: String,

    /// JSON Schema defining the tool's parameters.
    pub parameters: serde_json::Value,
}

/// A chunk from a streaming LLM response.
///
/// Mirrors the streaming event structure used in `victor.framework.events`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StreamChunk {
    /// Incremental text content in this chunk.
    pub content: Option<String>,

    /// Incremental tool call data in this chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Whether this is the final chunk in the stream.
    pub is_final: bool,

    /// Token usage statistics (typically only present in the final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = Message {
            role: "user".to_string(),
            content: Some("Hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));
        // Optional fields with None should be omitted
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("tool_call_id"));

        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_message_with_tool_calls() {
        let msg = Message {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "/tmp/test.txt"}),
            }]),
            tool_call_id: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);

        let tool_calls = deserialized.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "read_file");
    }

    #[test]
    fn test_tool_message() {
        let msg = Message {
            role: "tool".to_string(),
            content: Some("File contents here".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_123".to_string()),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"tool_call_id\":\"call_123\""));

        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_completion_response() {
        let resp = CompletionResponse {
            content: Some("Hello!".to_string()),
            tool_calls: None,
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
            model: "gpt-4".to_string(),
            stop_reason: Some("stop".to_string()),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: CompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, deserialized);
        assert_eq!(deserialized.usage.unwrap().total_tokens, 15);
    }

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file from disk".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read"
                    }
                },
                "required": ["path"]
            }),
        };

        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(tool, deserialized);
    }

    #[test]
    fn test_stream_chunk_intermediate() {
        let chunk = StreamChunk {
            content: Some("Hello".to_string()),
            tool_calls: None,
            is_final: false,
            usage: None,
        };

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("usage"));

        let deserialized: StreamChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk, deserialized);
        assert!(!deserialized.is_final);
    }

    #[test]
    fn test_stream_chunk_final() {
        let chunk = StreamChunk {
            content: None,
            tool_calls: None,
            is_final: true,
            usage: Some(Usage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
            }),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: StreamChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk, deserialized);
        assert!(deserialized.is_final);
        assert_eq!(deserialized.usage.unwrap().total_tokens, 150);
    }

    #[test]
    fn test_usage() {
        let usage = Usage {
            prompt_tokens: 42,
            completion_tokens: 13,
            total_tokens: 55,
        };

        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage, deserialized);
    }
}
