# Error recovery suggestion database

ERROR_RECOVERY_SUGGESTIONS = {
    # API Key Errors
    "api_key_missing": {
        "error_patterns": ["api key", "unauthorized", "401"],
        "suggestions": [
            "Set API key environment variable",
            "Check API key in profiles.yaml",
            "Verify API key has required permissions",
        ],
        "commands": [
            "export {PROVIDER}_API_KEY=your_key_here",
            "victor doctor --credentials",
            "victor profile show {profile_name}",
        ],
    },

    # Provider Connection Errors
    "provider_connection": {
        "error_patterns": ["connection refused", "connection timeout", "unreachable"],
        "suggestions": [
            "Check if provider service is running",
            "Verify network connectivity",
            "Check provider URL configuration",
        ],
        "commands": [
            "victor doctor --providers",
            "curl -I {provider_url}",
            "ping {provider_host}",
        ],
    },

    # Model Not Found Errors
    "model_not_found": {
        "error_patterns": ["model not found", "no such model", "model does not exist"],
        "suggestions": [
            "Check model name spelling",
            "Verify model is available for provider",
            "List available models for provider",
        ],
        "commands": [
            "victor models list --provider {provider_name}",
            "victor profile show {profile_name}",
            "{provider_cmd} list models",  # e.g., ollama list
        ],
    },

    # File System Errors
    "file_not_found": {
        "error_patterns": ["no such file", "file not found", "[errno 2]"],
        "suggestions": [
            "Check file path is correct",
            "Verify file exists in current directory",
            "Use absolute path if relative path fails",
        ],
        "commands": [
            "ls -la {file_path}",
            "pwd",
            "find . -name {filename}",
        ],
    },

    "permission_denied": {
        "error_patterns": ["permission denied", "[errno 13]", "access denied"],
        "suggestions": [
            "Check file/directory permissions",
            "Run with sudo if appropriate (use with caution)",
            "Verify you own the file/directory",
        ],
        "commands": [
            "ls -la {file_path}",
            "chmod +x {file_path}",
            "whoami",
        ],
    },

    # Network Errors
    "network_timeout": {
        "error_patterns": ["timeout", "timed out", "network unreachable"],
        "suggestions": [
            "Check internet connection",
            "Increase timeout setting",
            "Verify remote service is available",
        ],
        "commands": [
            "ping -c 3 {remote_host}",
            "curl -I {remote_url}",
            "victor config set timeout 600",
        ],
    },

    # Rate Limit Errors
    "rate_limit": {
        "error_patterns": ["rate limit", "429", "too many requests"],
        "suggestions": [
            "Wait before retrying",
            "Reduce request frequency",
            "Check if API tier needs upgrade",
        ],
        "commands": [
            "sleep 60",  # Wait 1 minute
            "victor doctor --usage",
        ],
    },

    # Docker Errors
    "docker_not_running": {
        "error_patterns": ["docker daemon", "docker not running", "connection refused to docker"],
        "suggestions": [
            "Start Docker daemon",
            "Check Docker service status",
            "Verify Docker is installed",
        ],
        "commands": [
            "docker ps",
            "sudo systemctl start docker",
            "docker info",
        ],
    },

    # Memory Errors
    "out_of_memory": {
        "error_patterns": ["out of memory", "oom", "memory", "cannot allocate"],
        "suggestions": [
            "Close other applications",
            "Reduce model size or context window",
            "Check system memory availability",
        ],
        "commands": [
            "free -h",  # Linux
            "vm_stat",  # macOS
            "victor config set max_tokens 2048",
        ],
    },

    # Configuration Errors
    "config_invalid": {
        "error_patterns": ["configuration error", "invalid config", "yaml error"],
        "suggestions": [
            "Validate configuration files",
            "Check YAML syntax",
            "Review profile configuration",
        ],
        "commands": [
            "victor config validate",
            "victor profile show {profile_name}",
            "cat ~/.victor/profiles.yaml",
        ],
    },

    # Tool Execution Errors
    "tool_execution_failed": {
        "error_patterns": ["tool execution failed", "tool error", "tool failed"],
        "suggestions": [
            "Check tool documentation",
            "Verify tool arguments",
            "Review tool permissions",
        ],
        "commands": [
            "victor tools list",
            "victor tools info {tool_name}",
            "victor doctor --tools",
        ],
    },
}


def get_recovery_suggestions(error_message: str, context: dict | None = None) -> dict:
    """Get recovery suggestions for an error.

    Args:
        error_message: The error message
        context: Additional context (provider, tool, file_path, etc.)

    Returns:
        Dictionary with error_type, suggestions, and commands
    """
    context = context or {}
    error_lower = error_message.lower()

    # Find matching error type
    for error_type, config in ERROR_RECOVERY_SUGGESTIONS.items():
        for pattern in config["error_patterns"]:
            if pattern in error_lower:
                # Replace placeholders in commands with context
                commands = []
                for cmd_template in config["commands"]:
                    cmd = cmd_template
                    # Replace {provider_name}, {profile_name}, etc. with context values
                    for key, value in context.items():
                        cmd = cmd.replace(f"{{{key.upper()}}}", str(value))
                    commands.append(cmd)

                return {
                    "error_type": error_type,
                    "suggestions": config["suggestions"],
                    "commands": commands,
                    "matched_pattern": pattern,
                }

    # Default fallback suggestions
    return {
        "error_type": "unknown",
        "suggestions": [
            "Run 'victor doctor' for diagnostics",
            "Check error details above",
            "Review configuration files",
        ],
        "commands": [
            "victor doctor",
            "victor config validate",
        ],
        "matched_pattern": None,
    }


def format_recovery_suggestions(recovery_data: dict) -> str:
    """Format recovery suggestions for display.

    Args:
        recovery_data: Dictionary from get_recovery_suggestions

    Returns:
        Formatted string for display
    """
    lines = []
    lines.append(f"Error Type: {recovery_data['error_type'].replace('_', ' ').title()}")
    lines.append("")

    lines.append("Suggestions:")
    for i, suggestion in enumerate(recovery_data["suggestions"], 1):
        lines.append(f"  {i}. {suggestion}")
    lines.append("")

    if recovery_data["commands"]:
        lines.append("Commands to try:")
        for cmd in recovery_data["commands"]:
            lines.append(f"  $ {cmd}")
        lines.append("")

    return "\n".join(lines)
