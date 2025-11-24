"""Demo of Victor's Testing Tool.

Demonstrates automated test generation:
- Generate unit tests for functions
- Generate tests for classes
- Create test fixtures
- Analyze test coverage gaps
- Generate mock data

Usage:
    python examples/testing_demo.py
"""

import asyncio
import tempfile
from pathlib import Path
from victor.tools.testing_tool import TestingTool


def setup_demo_file(temp_dir: Path, filename: str, content: str) -> Path:
    """Create a demo file."""
    file_path = temp_dir / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


async def demo_generate_function_tests():
    """Demo generating tests for functions."""
    print("\n\n🧪 Generate Function Tests Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create demo module
        demo_code = """
def calculate_total(items, tax_rate=0.08):
    \"\"\"Calculate total with tax.\"\"\"
    if not items:
        return 0

    subtotal = sum(item['price'] for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax


def validate_email(email):
    \"\"\"Validate email address.\"\"\"
    if not email or '@' not in email:
        raise ValueError("Invalid email")

    return email.lower()


def filter_active(users):
    \"\"\"Filter active users.\"\"\"
    return [u for u in users if u.get('active', False)]
"""
        file_path = setup_demo_file(temp_path, "calc.py", demo_code.strip())

        tool = TestingTool()

        print("\n1️⃣ Source code:")
        print(demo_code)

        print("\n2️⃣ Generate tests for all functions...")
        result = await tool.execute(
            operation="generate_tests",
            file=str(file_path),
            output=str(temp_path / "test_calc.py"),
        )

        if result.success:
            print(result.output)

        print("\n3️⃣ Generated test file:")
        test_file = temp_path / "test_calc.py"
        if test_file.exists():
            print(test_file.read_text())


async def demo_generate_class_tests():
    """Demo generating tests for classes."""
    print("\n\n🏗️  Generate Class Tests Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create demo module
        demo_code = """
class Calculator:
    \"\"\"Simple calculator.\"\"\"

    def __init__(self):
        self.memory = 0

    def add(self, a, b):
        \"\"\"Add two numbers.\"\"\"
        result = a + b
        self.memory = result
        return result

    def subtract(self, a, b):
        \"\"\"Subtract two numbers.\"\"\"
        result = a - b
        self.memory = result
        return result

    def clear(self):
        \"\"\"Clear memory.\"\"\"
        self.memory = 0


class UserManager:
    \"\"\"Manage users.\"\"\"

    def __init__(self, db):
        self.db = db
        self.users = []

    def add_user(self, user):
        \"\"\"Add a user.\"\"\"
        if not user.get('email'):
            raise ValueError("Email required")
        self.users.append(user)
        return user

    def get_user(self, user_id):
        \"\"\"Get user by ID.\"\"\"
        for user in self.users:
            if user['id'] == user_id:
                return user
        return None
"""
        file_path = setup_demo_file(temp_path, "models.py", demo_code.strip())

        tool = TestingTool()

        print("\n1️⃣ Source code:")
        print(demo_code[:500] + "...")

        print("\n2️⃣ Generate tests for classes...")
        result = await tool.execute(
            operation="generate_tests",
            file=str(file_path),
            output=str(temp_path / "test_models.py"),
        )

        if result.success:
            print(result.output)


async def demo_generate_fixture():
    """Demo generating test fixtures."""
    print("\n\n📦 Generate Test Fixture Demo")
    print("=" * 70)

    tool = TestingTool()

    print("\n1️⃣ Generate list fixture...")
    result = await tool.execute(
        operation="generate_fixture",
        name="sample_items",
        type="list",
    )

    if result.success:
        print(result.output)

    print("\n\n2️⃣ Generate dict fixture...")
    result = await tool.execute(
        operation="generate_fixture",
        name="sample_config",
        type="dict",
    )

    if result.success:
        print(result.output)

    print("\n\n3️⃣ Generate object fixture...")
    result = await tool.execute(
        operation="generate_fixture",
        name="mock_service",
        type="object",
    )

    if result.success:
        print(result.output)


async def demo_analyze_coverage():
    """Demo analyzing test coverage gaps."""
    print("\n\n📊 Analyze Coverage Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create complex module
        demo_code = """
def process_order(order, user):
    \"\"\"Process an order.\"\"\"
    if not order or not user:
        raise ValueError("Order and user required")

    if not user.get('verified'):
        raise PermissionError("User not verified")

    total = calculate_order_total(order)

    if total > user.get('credit_limit', 0):
        return {'status': 'declined', 'reason': 'Credit limit exceeded'}

    return {'status': 'approved', 'total': total}


def calculate_order_total(order):
    \"\"\"Calculate order total.\"\"\"
    subtotal = sum(item['price'] * item['quantity'] for item in order['items'])

    if order.get('discount_code'):
        discount = subtotal * 0.1
        subtotal -= discount

    shipping = 10.0 if subtotal < 100 else 0

    return subtotal + shipping


def send_notification(user, message, channel='email'):
    \"\"\"Send notification to user.\"\"\"
    if channel == 'email':
        send_email(user['email'], message)
    elif channel == 'sms':
        send_sms(user['phone'], message)
    else:
        raise ValueError(f"Unknown channel: {channel}")


def send_email(email, message):
    pass


def send_sms(phone, message):
    pass
"""
        file_path = setup_demo_file(temp_path, "orders.py", demo_code.strip())

        tool = TestingTool()

        print("\n1️⃣ Source code (complex business logic):")
        print(demo_code[:400] + "...")

        print("\n\n2️⃣ Analyze test coverage gaps...")
        result = await tool.execute(
            operation="analyze_coverage",
            file=str(file_path),
        )

        if result.success:
            print(result.output)


async def demo_scaffold():
    """Demo creating test file scaffold."""
    print("\n\n🏗️  Scaffold Test File Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create simple module
        demo_code = """
def greet(name):
    return f"Hello, {name}!"
"""
        file_path = setup_demo_file(temp_path, "app.py", demo_code.strip())

        tool = TestingTool()

        print("\n1️⃣ Create test scaffold...")
        result = await tool.execute(
            operation="scaffold",
            file=str(file_path),
            output=str(temp_path / "test_app.py"),
        )

        if result.success:
            print(result.output)


async def demo_mock_data():
    """Demo generating mock data."""
    print("\n\n🎭 Generate Mock Data Demo")
    print("=" * 70)

    tool = TestingTool()

    print("\n1️⃣ Generate mock list data...")
    result = await tool.execute(
        operation="mock_data",
        name="mock_users",
        type="list",
    )

    if result.success:
        print(result.output)

    print("\n\n2️⃣ Generate mock dict data...")
    result = await tool.execute(
        operation="mock_data",
        name="mock_config",
        type="dict",
    )

    if result.success:
        print(result.output)

    print("\n\n3️⃣ Generate mock user data...")
    result = await tool.execute(
        operation="mock_data",
        name="mock_user",
        type="user",
    )

    if result.success:
        print(result.output)


async def demo_real_world_workflow():
    """Demo a real-world testing workflow."""
    print("\n\n🎯 Real-World Testing Workflow Demo")
    print("=" * 70)
    print("\nScenario: Adding tests to existing code")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create realistic module
        demo_code = """
from typing import List, Dict, Optional


class ShoppingCart:
    \"\"\"Shopping cart for e-commerce.\"\"\"

    def __init__(self):
        self.items: List[Dict] = []
        self.discount_code: Optional[str] = None

    def add_item(self, product_id: int, name: str, price: float, quantity: int = 1):
        \"\"\"Add item to cart.\"\"\"
        if price < 0:
            raise ValueError("Price cannot be negative")

        if quantity < 1:
            raise ValueError("Quantity must be at least 1")

        self.items.append({
            'product_id': product_id,
            'name': name,
            'price': price,
            'quantity': quantity,
        })

    def remove_item(self, product_id: int):
        \"\"\"Remove item from cart.\"\"\"
        self.items = [item for item in self.items if item['product_id'] != product_id]

    def get_total(self) -> float:
        \"\"\"Calculate cart total.\"\"\"
        subtotal = sum(item['price'] * item['quantity'] for item in self.items)

        if self.discount_code == 'SAVE10':
            subtotal *= 0.9  # 10% off

        return round(subtotal, 2)

    def clear(self):
        \"\"\"Clear cart.\"\"\"
        self.items = []
        self.discount_code = None
"""
        file_path = setup_demo_file(temp_path, "cart.py", demo_code.strip())

        tool = TestingTool()

        print("\n1️⃣ Source code (ShoppingCart class):")
        print(demo_code[:500] + "...")

        print("\n\n2️⃣ STEP 1: Analyze coverage gaps...")
        result = await tool.execute(
            operation="analyze_coverage",
            file=str(file_path),
        )
        if result.success:
            print("✓ Coverage analysis complete")
            print(result.output[:500] + "...")

        print("\n\n3️⃣ STEP 2: Generate comprehensive test suite...")
        result = await tool.execute(
            operation="generate_tests",
            file=str(file_path),
            output=str(temp_path / "test_cart.py"),
        )
        if result.success:
            print("✓ Test suite generated")
            print("\nGenerated tests preview:")
            test_file = temp_path / "test_cart.py"
            if test_file.exists():
                content = test_file.read_text()
                print(content[:800] + "...")

        print("\n\n4️⃣ STEP 3: Generate test fixtures...")
        result = await tool.execute(
            operation="generate_fixture",
            name="sample_cart_items",
            type="list",
        )
        if result.success:
            print("✓ Fixtures generated")

        print("\n\n5️⃣ STEP 4: Generate mock data...")
        result = await tool.execute(
            operation="mock_data",
            name="mock_products",
            type="list",
        )
        if result.success:
            print("✓ Mock data generated")

        print("\n\n✅ Testing Workflow Complete!")
        print("\nWhat we accomplished:")
        print("  • Analyzed code for test coverage gaps")
        print("  • Generated comprehensive test suite")
        print("  • Created pytest fixtures")
        print("  • Generated mock data")
        print("  • Ready to run: pytest test_cart.py")


async def demo_generated_tests_example():
    """Show example of what generated tests look like."""
    print("\n\n📝 Generated Tests Example")
    print("=" * 70)

    example_test = '''"""Tests for utils.py."""

import pytest
from utils import *


# Fixtures

@pytest.fixture
def sample_data():
    """Sample test data."""
    return {
        "name": "Test User",
        "email": "test@example.com",
        "age": 25,
    }


# Function Tests

def test_validate_email():
    """Test validate_email function."""
    # Arrange
    email = "test@example.com"

    # Act
    result = validate_email(email)

    # Assert
    assert result is not None
    assert result == "test@example.com"


def test_validate_email_edge_cases():
    """Test validate_email with edge cases."""
    # Test with None
    with pytest.raises(ValueError):
        validate_email(None)

    # Test with empty string
    with pytest.raises(ValueError):
        validate_email("")

    # Test without @
    with pytest.raises(ValueError):
        validate_email("notanemail")


def test_validate_email_errors():
    """Test validate_email error handling."""
    with pytest.raises(ValueError):
        validate_email("invalid")


# Class Tests

class TestUserManager:
    """Tests for UserManager class."""

    @pytest.fixture
    def instance(self):
        """Create UserManager instance for testing."""
        return UserManager(db=None)

    def test_add_user(self, instance):
        """Test add_user method."""
        # Arrange
        user = {"id": 1, "email": "test@example.com"}

        # Act
        result = instance.add_user(user)

        # Assert
        assert result is not None
        assert result['email'] == "test@example.com"

    def test_get_user(self, instance):
        """Test get_user method."""
        # Arrange
        user = {"id": 1, "email": "test@example.com"}
        instance.add_user(user)

        # Act
        result = instance.get_user(1)

        # Assert
        assert result is not None
        assert result['id'] == 1
'''

    print("\n✨ Example of auto-generated test file:")
    print(example_test)
    print("\n\nFeatures of generated tests:")
    print("  • pytest-compatible structure")
    print("  • Fixture setup for test data")
    print("  • Arrange-Act-Assert pattern")
    print("  • Happy path tests")
    print("  • Edge case tests")
    print("  • Error handling tests")
    print("  • Class-based test organization")
    print("  • TODO comments for customization")


async def main():
    """Run all testing demos."""
    print("🎯 Victor Testing Tool Demo")
    print("=" * 70)
    print("\nDemonstrating automated test generation\n")

    # Run demos
    await demo_generate_function_tests()
    await demo_generate_class_tests()
    await demo_generate_fixture()
    await demo_analyze_coverage()
    await demo_scaffold()
    await demo_mock_data()
    await demo_real_world_workflow()
    await demo_generated_tests_example()

    print("\n\n✨ Demo Complete!")
    print("\nVictor's Testing Tool provides:")
    print("  • AST-based code analysis")
    print("  • pytest-compatible test generation")
    print("  • Fixture and mock data creation")
    print("  • Coverage gap analysis")
    print("  • Test scaffolding")
    print("  • Arrange-Act-Assert pattern")
    print("  • Edge case and error tests")
    print("\nPerfect for:")
    print("  • Adding tests to legacy code")
    print("  • TDD/BDD workflows")
    print("  • Increasing code coverage")
    print("  • Standardizing test structure")
    print("  • Rapid test suite development")
    print("\nGenerated tests are:")
    print("  • Customizable with TODO markers")
    print("  • Production-ready structure")
    print("  • Following best practices")
    print("\nReady to accelerate your testing!")


if __name__ == "__main__":
    asyncio.run(main())
