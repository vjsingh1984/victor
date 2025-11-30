#!/bin/bash

# Function to generate Fibonacci series
generate_fibonacci() {
    local n=$1
    
    # Check if input is valid
    if [[ ! $n =~ ^[0-9]+$ ]] || [ $n -le 0 ]; then
        echo "Error: Please provide a positive integer"
        return 1
    fi
    
    echo "Fibonacci series for n=$n:"
    
    # Handle special cases
    if [ $n -eq 1 ]; then
        echo "0"
        return 0
    elif [ $n -eq 2 ]; then
        echo "0 1"
        return 0
    fi
    
    # Initialize first two numbers
    local a=0
    local b=1
    
    # Print first two numbers
    echo -n "0 1"
    
    # Generate remaining numbers
    for ((i=2; i<n; i++)); do
        local next=$((a + b))
        echo -n " $next"
        a=$b
        b=$next
    done
    
    echo  # New line at the end
}

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_terms>"
    echo "Example: $0 10"
    exit 1
fi

# Generate Fibonacci series
generate_fibonacci $1