#!/bin/bash

# Generate Fibonacci series for n=5

# Initialize first two numbers
a=0
b=1

# Print the first two numbers
if [ $1 -eq 1 ]; then
    echo "$a"
elif [ $1 -eq 2 ]; then
    echo "$a $b"
else
    echo -n "$a $b "
    
    # Generate remaining numbers
    for ((i=2; i<5; i++)); do
        next=$((a + b))
        echo -n "$next "
        a=$b
        b=$next
    done
    echo ""
fi

# Generate the series for n=5 specifically
if [ $# -eq 0 ]; then
    echo "Fibonacci series for n=5:"
    echo -n "0 1 "
    a=0
    b=1
    for ((i=2; i<5; i++)); do
        next=$((a + b))
        echo -n "$next "
        a=$b
        b=$next
    done
    echo ""
fi

# Alternative simpler approach for n=5
if [ $# -eq 0 ]; then
    echo "Fibonacci series for n=5:"
    echo "0 1 1 2 3"
fi
