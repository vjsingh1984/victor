def fibonacci(n):
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n (int): The position in the Fibonacci sequence (non-negative integer)
        
    Returns:
        int: The nth Fibonacci number
        
    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(5)
        5
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

# Alternative recursive implementation (less efficient for large numbers)
def fibonacci_recursive(n):
    """
    Calculate the nth Fibonacci number using recursion.
    
    Args:
        n (int): The position in the Fibonacci sequence (non-negative integer)
        
    Returns:
        int: The nth Fibonacci number
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Test the function
if __name__ == "__main__":
    print("First 10 Fibonacci numbers:")
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")