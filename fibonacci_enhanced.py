def fibonacci(n):
    """
    Calculate the nth Fibonacci number using an iterative approach.
    
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


def fibonacci_memoized(n, memo=None):
    """
    Calculate the nth Fibonacci number using memoization.
    
    Args:
        n (int): The position in the Fibonacci sequence (non-negative integer)
        memo (dict): Memoization dictionary (default: None)
        
    Returns:
        int: The nth Fibonacci number
    """
    if memo is None:
        memo = {}
    
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    
    if n in memo:
        return memo[n]
    
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
        return memo[n]


def fibonacci_generator(limit):
    """
    Generate Fibonacci numbers up to a limit.
    
    Args:
        limit (int): Maximum number of Fibonacci numbers to generate
        
    Yields:
        int: Next Fibonacci number in the sequence
    """
    if limit <= 0:
        return
    
    a, b = 0, 1
    count = 0
    
    while count < limit:
        if count == 0:
            yield a
        elif count == 1:
            yield b
        else:
            a, b = b, a + b
            yield b
        count += 1


def fibonacci_list(n):
    """
    Generate a list of first n Fibonacci numbers.
    
    Args:
        n (int): Number of Fibonacci numbers to generate
        
    Returns:
        list: List containing first n Fibonacci numbers
    """
    if n <= 0:
        return []
    
    result = []
    for i in range(n):
        result.append(fibonacci(i))
    
    return result


# Test the functions
if __name__ == "__main__":
    print("First 10 Fibonacci numbers (iterative):")
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")
    
    print("\nFirst 10 Fibonacci numbers (memoized):")
    for i in range(10):
        print(f"F({i}) = {fibonacci_memoized(i)}")
    
    print("\nFirst 10 Fibonacci numbers (generator):")
    fib_gen = fibonacci_generator(10)
    for i, num in enumerate(fib_gen):
        print(f"F({i}) = {num}")
    
    print("\nList of first 10 Fibonacci numbers:")
    print(fibonacci_list(10))