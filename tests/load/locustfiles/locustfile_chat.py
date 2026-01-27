# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Locust load test file for Victor AI chat conversations.

This file simulates realistic multi-turn chat conversations.
Run with: locust -f tests/load/locustfiles/locustfile_chat.py

Or headless: locust -f tests/load/locustfiles/locustfile_chat.py --headless --users 50 --spawn-rate 5 --run-time 10m
"""

import random
import time
from datetime import datetime
from typing import List, Dict

from locust import HttpUser, task, between, events


class ChatUser(HttpUser):
    """Simulates realistic multi-turn chat conversations.

    User Behavior:
    - Starts a conversation
    - Has 5-15 turn conversation
    - Ends conversation and starts new one
    - Simulates realistic thinking time between messages
    """

    # Wait between messages: 2-10 seconds (realistic conversation pace)
    wait_time = between(2, 10)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider = "anthropic"
        self.model = "claude-sonnet-4-5"
        self.conversation_history: List[Dict] = []
        self.turn_count = 0
        self.max_turns = random.randint(5, 15)

        # Select conversation topic
        self.topic = random.choice(
            [
                "coding_help",
                "debugging",
                "code_review",
                "learning",
                "explanation",
            ]
        )

    def on_start(self):
        """Start new conversation."""
        self.start_new_conversation()

    def start_new_conversation(self):
        """Initialize a new conversation."""
        self.conversation_history = []
        self.turn_count = 0
        self.max_turns = random.randint(5, 15)
        self.topic = random.choice(
            [
                "coding_help",
                "debugging",
                "code_review",
                "learning",
                "explanation",
            ]
        )

    def get_next_message(self) -> str:
        """Generate next message based on conversation state."""
        self.turn_count += 1

        # Coding help conversation
        if self.topic == "coding_help":
            messages = [
                "I need help writing a Python function",
                "How do I handle exceptions in it?",
                "Can you add type hints?",
                "What about docstrings?",
                "How do I test this function?",
                "Can you optimize it?",
                "What's the time complexity?",
                "Can you add logging?",
                "How do I handle edge cases?",
                "Thanks, that helps!",
            ]

        # Debugging conversation
        elif self.topic == "debugging":
            messages = [
                "I have a bug in my code",
                "Here's the error message",
                "It happens when I run the tests",
                "Only in production, not locally",
                "Can you help me fix it?",
                "What's the root cause?",
                "How do I prevent this?",
                "Can you add unit tests?",
            ]

        # Code review conversation
        elif self.topic == "code_review":
            messages = [
                "Can you review this code?",
                "What about security issues?",
                "Is it readable?",
                "How's the performance?",
                "Any best practices I'm missing?",
                "Can you suggest refactoring?",
                "What about error handling?",
                "Should I add more tests?",
            ]

        # Learning conversation
        elif self.topic == "learning":
            messages = [
                "Can you explain async/await?",
                "How does it differ from threads?",
                "When should I use it?",
                "Can you show me an example?",
                "What are common pitfalls?",
                "How do I debug async code?",
                "What about async generators?",
                "Can you explain the event loop?",
            ]

        # Explanation conversation
        else:  # explanation
            messages = [
                "What is a REST API?",
                "How does HTTP work?",
                "What are the different methods?",
                "Can you explain status codes?",
                "What about authentication?",
                "How do I design a good API?",
                "What's RESTful vs GraphQL?",
                "Can you show me examples?",
            ]

        # Get message based on turn count
        index = (self.turn_count - 1) % len(messages)
        return messages[index]

    @task
    def chat_turn(self):
        """Send a message in the conversation."""
        # Check if conversation should end
        if self.turn_count >= self.max_turns:
            self.start_new_conversation()

        message = self.get_next_message()

        # Build messages list with conversation history
        messages = []
        for msg in self.conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # Add current message
        messages.append({"role": "user", "content": message})

        payload = {"messages": messages}

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat",
        ) as response:
            if response.status_code == 200:
                try:
                    # Record response in conversation history
                    self.conversation_history.append(
                        {
                            "role": "user",
                            "content": message,
                        }
                    )
                    self.conversation_history.append(
                        {
                            "role": "assistant",
                            "content": response.text[:200],  # Truncate for memory
                        }
                    )
                    response.success()
                except Exception as e:
                    response.failure(f"Failed to process response: {e}")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(3)
    def chat_turn_with_tool(self):
        """Send a message that might use tools."""
        # Check if conversation should end
        if self.turn_count >= self.max_turns:
            self.start_new_conversation()

        # Tool-using messages
        tool_messages = [
            "Read the file README.md",
            "List all Python files in the current directory",
            "Search for 'TODO' comments in the codebase",
            "Check the git status",
            "What tests are defined?",
            "Find all functions named 'test'",
            "Analyze the code structure",
            "Count the lines of code",
            "Check for syntax errors",
            "What Python version is installed?",
        ]

        message = random.choice(tool_messages)

        # Build messages list with conversation history
        messages = []
        for msg in self.conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # Add current message
        messages.append({"role": "user", "content": message})

        payload = {"messages": messages}

        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat (with tool)",
        ) as response:
            if response.status_code == 200:
                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": message,
                    }
                )
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


class QuickChatUser(ChatUser):
    """User that has short, quick conversations."""

    wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = random.randint(2, 5)  # Short conversations


class LongChatUser(ChatUser):
    """User that has long, detailed conversations."""

    wait_time = between(5, 15)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = random.randint(15, 30)  # Long conversations


class CodingSessionUser(ChatUser):
    """User simulating a coding session with frequent tool use."""

    wait_time = between(3, 8)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topic = "coding_help"
        self.max_turns = random.randint(10, 20)

    @task(5)
    def chat_turn_with_tool(self):
        """Override to use tools more frequently."""
        super().chat_turn_with_tool()

    @task(1)
    def chat_turn(self):
        """Regular chat turns less frequently."""
        super().chat_turn()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test summary."""
    print("\n" + "=" * 80)
    print("VICTOR AI CHAT CONVERSATION LOAD TEST SUMMARY")
    print("=" * 80)
    print(f"Test completed at: {datetime.now().isoformat()}")

    if environment.stats:
        stats = environment.stats

        print(f"\nTotal Requests: {stats.total.num_requests}")
        print(f"Successful: {stats.total.num_requests - stats.total.num_failures}")
        print(f"Failed: {stats.total.num_failures}")

        if stats.total.num_requests > 0:
            print(
                f"Success Rate: {((stats.total.num_requests - stats.total.num_failures) / stats.total.num_requests * 100):.2f}%"
            )

        print("\nResponse Times:")
        print(f"  Median: {stats.total.median_response_time:.0f}ms")
        print(f"  95th percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")
        print(f"  99th percentile: {stats.total.get_response_time_percentile(0.99):.0f}ms")

        print(f"\nThroughput: {stats.total.total_avg_rps:.2f} requests/second")

    print("=" * 80 + "\n")
