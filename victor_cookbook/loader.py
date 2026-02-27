"""Recipe loader for discovering and loading Victor cookbook recipes."""

from pathlib import Path
from typing import Dict, List, Optional
import importlib.util
import sys


class RecipeLoader:
    """Load and manage cookbook recipes."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize recipe loader.

        Args:
            base_path: Base path for recipes (default: package recipes/ directory)
        """
        if base_path is None:
            base_path = Path(__file__).parent / "recipes"
        self.base_path = Path(base_path)

    def list_recipes(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """List all available recipes by category.

        Args:
            category: Filter by category (None for all)

        Returns:
            Dictionary mapping category to list of recipe names
        """
        recipes = {}

        for cat_dir in self.base_path.iterdir():
            if not cat_dir.is_dir():
                continue

            cat_name = cat_dir.name
            if category and cat_name != category:
                continue

            recipes[cat_name] = []

            for recipe_file in cat_dir.glob("*.py"):
                if recipe_file.name.startswith("_"):
                    continue

                recipes[cat_name].append(recipe_file.stem)

        return recipes

    def load_recipe(self, category: str, recipe_name: str) -> Dict:
        """Load a recipe module.

        Args:
            category: Recipe category (e.g., "agents/basic")
            recipe_name: Name of recipe (without .py extension)

        Returns:
            Recipe module dictionary with metadata and functions
        """
        recipe_path = self.base_path / category / f"{recipe_name}.py"

        if not recipe_path.exists():
            raise FileNotFoundError(f"Recipe not found: {category}/{recipe_name}")

        spec = importlib.util.spec_from_file_location(
            f"victor_cookbook.recipes.{category}.{recipe_name}",
            recipe_path
        )

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"victor_cookbook.recipes.{category}.{recipe_name}"] = module
        spec.loader.exec_module(module)

        # Extract recipe metadata
        return {
            "module": module,
            "name": getattr(module, "RECIPE_NAME", recipe_name),
            "description": getattr(module, "RECIPE_DESCRIPTION", ""),
            "category": category,
            "tags": getattr(module, "RECIPE_TAGS", []),
            "difficulty": getattr(module, "RECIPE_DIFFICULTY", "beginner"),
            "estimated_time": getattr(module, "RECIPE_TIME", "5 minutes"),
        }

    def search_recipes(self, query: str) -> List[Dict]:
        """Search for recipes by tag, name, or description.

        Args:
            query: Search query string

        Returns:
            List of matching recipe metadata dictionaries
        """
        results = []
        query_lower = query.lower()

        for category, recipes in self.list_recipes().items():
            for recipe_name in recipes:
                try:
                    metadata = self.load_recipe(category, recipe_name)
                except Exception:
                    continue

                # Search in name, description, tags
                if (
                    query_lower in metadata["name"].lower() or
                    query_lower in metadata["description"].lower() or
                    any(query_lower in tag.lower() for tag in metadata["tags"])
                ):
                    results.append(metadata)

        return results

    def get_recipe_info(self, category: str, recipe_name: str) -> str:
        """Get formatted information about a recipe.

        Args:
            category: Recipe category
            recipe_name: Recipe name

        Returns:
            Formatted recipe information
        """
        metadata = self.load_recipe(category, recipe_name)

        return f"""
Recipe: {metadata['name']}
Category: {metadata['category']}
Description: {metadata['description']}
Difficulty: {metadata['difficulty']}
Estimated Time: {metadata['estimated_time']}
Tags: {', '.join(metadata['tags'])}

Usage:
    from victor_cookbook.recipes.{category} import {recipe_name}
    result = await {recipe_name}.main()
"""


# Singleton instance
_recipe_loader = None


def get_loader() -> RecipeLoader:
    """Get the singleton recipe loader instance.

    Returns:
        RecipeLoader instance
    """
    global _recipe_loader
    if _recipe_loader is None:
        _recipe_loader = RecipeLoader()
    return _recipe_loader
