from typing import List


class Fibonacci:
    def calculate(self, n: int) -> int:
        """
        Calculate the nth Fibonacci number using iterative approach.

        Args:
            n (int): The position in the Fibonacci sequence

        Returns:
            int: The nth Fibonacci number
        """
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

    def sequence(self, n: int) -> List[int]:
        """
        Generate the Fibonacci sequence up to the nth number.

        Args:
            n (int): The number of elements in the sequence

        Returns:
            list: The Fibonacci sequence up to the nth number
        """
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
        else:
            fib_seq = [0, 1]
            for i in range(2, n):
                fib_seq.append(fib_seq[i - 1] + fib_seq[i - 2])
            return fib_seq


# Example usage
if __name__ == "__main__":
    fib = Fibonacci()
    n = 5
    result = fib.calculate(n)
    sequence = fib.sequence(n)
    print(f"The {n}th Fibonacci number is: {result}")
    print(f"Fibonacci sequence up to {n}: {sequence}")
