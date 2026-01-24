#!/bin/bash
# Comprehensive script to fix all missing type parameters in Victor codebase

cd /Users/vijaysingh/code/codingagent

echo "Fixing type parameters in Python files..."

# Fix Callable without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: Callable\([,)\]\]/: Callable[..., Any]\1/g' \
  -e 's/Dict\[str, Callable\]/Dict[str, Callable[..., Any]]/g' \
  -e 's/dict\[str, Callable\]/dict[str, Callable[..., Any]]/g' \
  {} \;

# Fix Dict/Dict without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: Dict\([,)\]\]/: Dict[str, Any]\1/g' \
  -e 's/: dict\([,)\]\]/: dict[str, Any]\1/g' \
  {} \;

# Fix List/list without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: List\([,)\]\]/: List[Any]\1/g' \
  -e 's/: list\([,)\]\]/: list[Any]\1/g' \
  {} \;

# Fix Tuple/tuple without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: Tuple\([,)\]\]/: Tuple[Any, ...]\1/g' \
  -e 's/: tuple\([,)\]\]/: tuple[Any, ...]\1/g' \
  {} \;

# Fix Set/set without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: Set\([,)\]\]/: Set[Any]\1/g' \
  -e 's/: set\([,)\]\]/: set[Any]\1/g' \
  {} \;

# Fix frozenset without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: frozenset\([,)\]\]/: frozenset[Any]\1/g' \
  {} \;

# Fix Type without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: Type\([,)\]\]/: Type[Any]\1/g' \
  {} \;

# Fix Pattern without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: Pattern\([,)\]\]/: Pattern[str]\1/g' \
  {} \;

# Fix Counter without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: Counter\([,)\]\]/: Counter[str]\1/g' \
  {} \;

# Fix deque without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: deque\([,)\]\]/: deque[Any]\1/g' \
  {} \;

# Fix OrderedDict without parameters
find victor -name "*.py" -type f -exec sed -i '' \
  -e 's/: OrderedDict\([,)\]\]/: OrderedDict[str, Any]\1/g' \
  {} \;

echo "Done. Verifying fixes..."

# Count remaining errors
ERRORS=$(mypy victor/ --config-file pyproject.toml 2>&1 | grep "type-arg" | wc -l | tr -d ' ')
echo "Remaining type-arg errors: $ERRORS"
