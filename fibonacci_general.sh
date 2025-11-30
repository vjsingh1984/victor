#!/bin/bash

# Function to generate Fibonacci series for n numbers
# Usage: fibonacci <n>
fibonacci() {
    local n=$1
    
    # Validate input
    if [[ ! $n =~ ^[0-9]+$ ]] || [[ $n -le 0 ]]; then
        echo "Error: Please provide a positive integer"
        return 1
    fi
    
    # Handle special cases
    if [[ $n -eq 1 ]]; then
        echo "Fibonacci series for n=$n:"
        echo "0"
        return 0
    fi
    
    if [[ $n -eq 2 ]]; then
        echo "Fibonacci series for n=$n:"
        echo "0 1"
        return 0
    fi
    
    # Generate Fibonacci series
    echo "Fibonacci series for n=$n:"
    local a=0
    local b=1
    echo -n "$a $b "
    
    # Generate remaining numbers
    for ((i=2; i<n; i++)); do
        local next=$((a + b))
        echo -n "$next "
        a=$b
        b=$next
    done
    
    echo ""  # New line at the end
}

# Check if argument is provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <number_of_terms>"
    echo "Example: $0 5"
    exit 1
fi

# Generate Fibonacci series
fibonacci $1