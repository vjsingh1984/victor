def fibonacci(n):
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n (int): The position of the Fibonacci number to calculate.
    
    Returns:
        int: The nth Fibonacci number.
    """
    if n <= 1:
        return n
    else:
        return(fibonacci(n-1) + fibonacci(n-2))