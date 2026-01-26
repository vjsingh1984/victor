"""Well-written Python code following best practices."""

import logging
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    """Enumeration for status values."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


@dataclass
class User:
    """User data model with type hints."""

    id: int
    name: str
    email: str
    status: Status


class UserService:
    """Service class for user operations with proper error handling."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize service with dependency injection."""
        self.logger = logger or logging.getLogger(__name__)

    def get_active_users(self, users: List[User]) -> List[User]:
        """Filter active users from the list.

        Args:
            users: List of user objects

        Returns:
            List of active users
        """
        return [user for user in users if user.status == Status.ACTIVE]

    def find_user_by_id(self, users: List[User], user_id: int) -> Optional[User]:
        """Find user by ID with proper error handling.

        Args:
            users: List of user objects
            user_id: User ID to search for

        Returns:
            User object if found, None otherwise
        """
        try:
            for user in users:
                if user.id == user_id:
                    return user
        except Exception as e:
            self.logger.error(f"Error finding user: {e}")
            return None

        return None


def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate discounted price with input validation.

    Args:
        price: Original price
        discount_percent: Discount percentage (0-100)

    Returns:
        Discounted price

    Raises:
        ValueError: If inputs are invalid
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")

    discount_amount = price * (discount_percent / 100)
    return round(price - discount_amount, 2)


def main():
    """Main function demonstrating proper usage."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create service
    service = UserService()

    # Create test users
    users = [
        User(1, "Alice", "alice@example.com", Status.ACTIVE),
        User(2, "Bob", "bob@example.com", Status.INACTIVE),
        User(3, "Charlie", "charlie@example.com", Status.ACTIVE),
    ]

    # Get active users
    active_users = service.get_active_users(users)
    print(f"Active users: {len(active_users)}")

    # Calculate discount
    discounted_price = calculate_discount(100.0, 20.0)
    print(f"Discounted price: ${discounted_price}")


if __name__ == "__main__":
    main()
