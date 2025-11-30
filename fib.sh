#!/bin/bash

# Function to generate Fibonacci series for n=5
fibonacci() {
    local n=5
    local a=0
    local b=1
    
    echo "Fibonacci series for n=5:"
    
    if [ $n -ge 1 ]; then
        echo -n "$a "
    fi
    
    if [ $n -ge 2 ]; then
        echo -n "$b "
    fi
    
    # Generate remaining numbers
    for ((i=3; i<=n; i++)); do
        local next=$((a + b))
        echo -n "$next "
        a=$b
        b=$next
    done
    
    echo ""  # New line at the end
}

# Call the function
fibonacci
