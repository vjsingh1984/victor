"""Shopping cart module with dataclass-based items."""

from dataclasses import dataclass
from typing import List


@dataclass
class ShoppingCartItem:
    """Represents an item in the shopping cart.

    Attributes:
        name: Name of the item
        price: Price per unit
        quantity: Number of units
    """
    name: str
    price: float
    quantity: int


class ShoppingCart:
    """Shopping cart for managing items and calculating totals.

    Attributes:
        items: List of ShoppingCartItem objects in the cart
    """

    def __init__(self):
        """Initialize an empty shopping cart."""
        self.items: List[ShoppingCartItem] = []

    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        """Add an item to the shopping cart.

        Args:
            name: Name of the item
            price: Price per unit
            quantity: Number of units to add (default: 1)
        """
        if price < 0:
            raise ValueError("Price cannot be negative")
        if quantity < 1:
            raise ValueError("Quantity must be at least 1")

        item = ShoppingCartItem(name=name, price=price, quantity=quantity)
        self.items.append(item)

    def remove_item(self, name: str) -> None:
        """Remove all instances of an item from the cart.

        Args:
            name: Name of the item to remove
        """
        self.items = [item for item in self.items if item.name != name]

    def get_total(self) -> float:
        """Calculate the total price of all items in the cart.

        Returns:
            Total price rounded to 2 decimal places
        """
        total = sum(item.price * item.quantity for item in self.items)
        return round(total, 2)

    def apply_discount(self, percent: float) -> float:
        """Calculate price after applying a percentage discount.

        Args:
            percent: Discount percentage (e.g., 10 for 10% off)

        Returns:
            Discounted total price
        """
        if percent < 0 or percent > 100:
            raise ValueError("Discount percent must be between 0 and 100")

        total = self.get_total()
        discount = (total / 100) * percent
        discounted_total = round(total - discount, 2)
        print(f'Applied {percent}% discount. Total: ${total:.2f} -> ${discounted_total:.2f}')
        return discounted_total

    def get_items(self) -> List[ShoppingCartItem]:
        """Get all items in the cart.

        Returns:
            List of ShoppingCartItem objects
        """
        return self.items


def main():
    """Example usage of the ShoppingCart class."""
    cart = ShoppingCart()

    # Add items
    cart.add_item('Apple', 1.00, 5)
    cart.add_item('Banana', 0.50, 10)
    cart.add_item('Orange', 2.00, 7)

    print(f'Total price before discount: ${cart.get_total()}')
    cart.apply_discount(10)

    print('\nItems in cart:')
    for item in cart.get_items():
        print(f'  {item.name}: {item.quantity} x ${item.price:.2f} = ${item.price * item.quantity:.2f}')


if __name__ == '__main__':
    main()
