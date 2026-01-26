"""Sample code demonstrating security vulnerabilities."""

import hashlib
import pickle
import subprocess


# Hardcoded credentials
API_SECRET = "super_secret_key_123"
DB_PASSWORD = "admin123"


def authenticate(username, password):
    """Insecure authentication function."""
    # Hardcoded password comparison
    if username == "admin" and password == "password123":
        return True
    return False


def execute_query(user_input):
    """SQL injection vulnerability."""
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    # Direct string concatenation allows SQL injection
    return query


def process_command(cmd):
    """Command injection vulnerability."""
    # Direct execution of user input
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout


def hash_password(password):
    """Weak hashing algorithm."""
    # MD5 is cryptographically broken
    return hashlib.md5(password.encode()).hexdigest()


def deserialize_data(data):
    """Insecure deserialization."""
    # Pickle deserialization can execute arbitrary code
    return pickle.loads(data)


# XSS vulnerability
def render_template(user_input):
    """Unescaped user input in template."""
    html = f"<div>{user_input}</div>"
    return html


# Path traversal vulnerability
def read_file(filename):
    """No path validation."""
    with open(filename, "r") as f:
        return f.read()


# Weak random number generation
import random


def generate_token():
    """Weak random token."""
    return random.randint(1000, 9999)


# No input validation
def calculate_discount(price, discount):
    """No validation of inputs."""
    return price * discount / 100
