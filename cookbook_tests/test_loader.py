"""Tests for RecipeLoader."""
import pytest
from victor_cookbook.loader import RecipeLoader


def test_loader_initialization():
    """Test that RecipeLoader can be initialized."""
    loader = RecipeLoader()
    assert loader is not None


def test_list_recipes():
    """Test that list_recipes returns a dictionary."""
    loader = RecipeLoader()
    recipes = loader.list_recipes()
    assert isinstance(recipes, dict)


def test_list_recipes_has_categories():
    """Test that list_recipes returns expected categories."""
    loader = RecipeLoader()
    recipes = loader.list_recipes()

    expected_categories = [
        "agents/basic",
        "agents/production",
        "agents/specialized",
        "workflows/automation",
        "workflows/decision_making",
        "workflows/data_processing",
        "integrations/apis",
        "integrations/databases",
        "integrations/messaging",
        "integrations/cloud",
    ]

    for category in expected_categories:
        assert category in recipes
        assert isinstance(recipes[category], list)


def test_search_recipes():
    """Test that search_recipes returns matching recipes."""
    loader = RecipeLoader()
    results = loader.search_recipes("slack")
    assert isinstance(results, list)
    # Should find Slack-related recipes
    assert len(results) > 0


def test_search_recipes_no_match():
    """Test search with no matches."""
    loader = RecipeLoader()
    results = loader.search_recipes("xyznonexistent")
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_load_recipe():
    """Test loading a specific recipe."""
    loader = RecipeLoader()
    recipe = loader.load_recipe("agents/basic", "simple_qa")

    assert "name" in recipe
    assert "category" in recipe
    assert "description" in recipe
    assert "function" in recipe
    assert recipe["name"] == "simple_qa"
    assert recipe["category"] == "agents/basic"


def test_load_recipe_not_found():
    """Test loading a non-existent recipe."""
    loader = RecipeLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_recipe("nonexistent", "category")


def test_recipe_count():
    """Test that we have the expected number of recipes."""
    loader = RecipeLoader()
    recipes = loader.list_recipes()

    total_count = sum(len(recipes) for recipes in recipes.values())
    # We should have at least 100 recipes
    assert total_count >= 100, f"Expected at least 100 recipes, found {total_count}"
