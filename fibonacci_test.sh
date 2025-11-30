#!/bin/bash

# Generate Fibonacci series for n=5
echo "Fibonacci series for n=5:"
a=0
b=1
echo -n "$a $b "

for ((i=2; i<5; i++)); do
    next=$((a + b))
    echo -n "$next "
    a=$b
    b=$next
done
echo ""