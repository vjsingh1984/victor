#!/bin/bash

# Generate Fibonacci series for n=3

echo "Fibonacci series for n=3:"

# First two Fibonacci numbers
a=0
b=1

echo -n "$a $b"

# Generate third number
next=$((a + b))
echo -n " $next"

echo  # New line at the end
