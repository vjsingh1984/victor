"""Sample Python code with various issues for review demonstration."""


def process_data(data_list, user_input):
    """Process data with some complexity issues."""
    results = []

    # Complex nested logic (high complexity)
    for item in data_list:
        if item is not None:
            if isinstance(item, dict):
                if 'value' in item:
                    if item['value'] > 0:
                        if item['value'] < 100:
                            for i in range(item['value']):
                                if i % 2 == 0:
                                    results.append(i * 2)
                                else:
                                    results.append(i)
                    elif item['value'] == 0:
                        results.append(0)
            elif isinstance(item, list):
                for sub_item in item:
                    if sub_item:
                        results.append(sub_item)

    # Very long line that exceeds standard line length limits and should be broken up into multiple lines for better readability and maintainability according to PEP 8 guidelines which recommend lines no longer than 79 characters
    final_result = sum(results) + len(data_list) + len(results) + len(user_input) + 100

    # Trailing whitespace
    user_query = user_input.strip()

    return final_result


class DataProcessor:
    """Data processing class with missing docstrings."""

    def __init__(self, config):
        self.config = config
        self.cache = {}

    def process(self, data):
        if not data:
            return None
        return [x for x in data if x]


# Hardcoded secret (security issue)
API_KEY = "sk-1234567890abcdef"
DATABASE_URL = "postgres://user:password@localhost/db"

# SQL injection vulnerability
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)


# Print statement instead of logger
def helper_function(x):
    print(f"Processing: {x}")  # Should use logger
    return x * 2
