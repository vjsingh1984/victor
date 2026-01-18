# This script creates a comprehensive set of sed commands to fix all 20 failing tests
# Each test follows a pattern that can be systematically replaced

# The general pattern to find and replace:
# async def <name>(<params>):
#     <body>
#     return await <call>
# result = benchmark.pedantic(<name>, <iters>, <rounds>)
# <var> = result[0]

# Should become:
# <var> = await benchmark(<call_without_return>)

# Due to the complexity of multi-line patterns in sed, this script outputs
# Python code that will perform the transformation correctly.

import re

# Read the file
with open("tests/benchmarks/test_tool_selection_performance.py", "r") as f:
    lines = f.readlines()

output = []
i = 0

while i < len(lines):
    line = lines[i]

    # Check for async def wrapper function pattern
    if re.match(r"\s+async def \w+\([^)]*\):", line) and i < len(lines) - 5:
        func_name = re.search(r"async def (\w+)\(", line).group(1)

        # Look ahead to see if this uses benchmark.pedantic
        j = i + 1
        found_benchmark = False
        func_body = []

        while j < len(lines) and not (
            lines[j].strip()
            and not lines[j].startswith("    ")
            and not lines[j].startswith("        ")
        ):
            func_body.append(lines[j])
            if "benchmark.pedantic(" in lines[j] and func_name in lines[j]:
                found_benchmark = True
                break
            j += 1

        if found_benchmark:
            # Extract the actual async call from the function body
            for k, body_line in enumerate(func_body):
                if "return await" in body_line or "return  await" in body_line:
                    # This is the line we need - extract the call
                    call_match = re.search(r"return\s+await\s+(.+)", body_line)

                    if call_match:
                        call = call_match.group(1).rstrip()

                        # The call might span multiple lines - collect them
                        full_call = [call]
                        m = k + 1
                        while m < len(func_body):
                            next_line = func_body[m].strip()
                            if next_line and not next_line.startswith("#"):
                                if ")" in next_line:
                                    full_call.append(next_line)
                                    break
                                full_call.append(next_line)
                            m += 1

                        call_str = " ".join(full_call).rstrip(",").rstrip()

                        # Parse the call to extract function and parameters
                        # Format: function_name(param1, param2, ...)
                        call_parts = re.match(r"(\w+)\.(\w+)\((.*)\)", call_str)

                        if call_parts:
                            obj_name = call_parts.group(1)
                            method_name = call_parts.group(2)
                            params = call_parts.group(3)

                            # Find the variable assignment (result[0])
                            var_name = None
                            for n in range(j, min(j + 3, len(lines))):
                                if "result[0]" in lines[n]:
                                    var_match = re.search(r"(\w+)\s*=\s*result\[0\]", lines[n])
                                    if var_match:
                                        var_name = var_match.group(1)
                                    break

                            if var_name:
                                # Generate the replacement code
                                output.append(f"    {var_name} = await benchmark(\n")
                                output.append(f"        {obj_name}.{method_name},\n")

                                # Add parameters with proper formatting
                                for param in params.split(","):
                                    param = param.strip()
                                    if param:
                                        if "=" in param:
                                            # Named parameter - keep as is
                                            output.append(f"        {param},\n")
                                        else:
                                            # Positional parameter - need to be more careful
                                            output.append(
                                                f"        {param},  # TODO: verify parameter name\n"
                                            )

                                output.append("    )\n")

                                # Skip all the lines we just processed
                                i = j + 1
                                while i < len(lines) and "result[0]" not in lines[i]:
                                    i += 1
                                i += 1  # Skip the result[0] line too
                                continue

    output.append(line)
    i += 1

# Write the fixed version
with open("tests/benchmarks/test_tool_selection_performance.py", "w") as f:
    f.writelines(output)

print("Fixed benchmark tests successfully!")
