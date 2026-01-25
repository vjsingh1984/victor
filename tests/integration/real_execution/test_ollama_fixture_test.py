# Simple test to verify Ollama fixture works
import pytest

@pytest.mark.asyncio
async def test_ollama_fixture_works(ollama_provider):
    """Test that Ollama fixture works."""
    from victor.providers.base import Message
    
    response = await ollama_provider.chat(
        messages=[Message(role="user", content="What is 2+2? Just say the number.")],
        model=ollama_provider._selected_model,
        max_tokens=10
    )
    
    assert response is not None
    assert response.content is not None
    assert "4" in response.content.lower() or "four" in response.content.lower()
    print(f"✓ Ollama fixture works! Response: {response.content}")
