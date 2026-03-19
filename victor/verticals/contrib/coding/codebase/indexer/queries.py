# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tree-sitter query dictionaries for multi-language symbol extraction.

These hardcoded dictionaries are LEGACY and will be DEPRECATED.
The new plugin-based architecture (victor/languages/plugins/) provides
these queries via TreeSitterQueries in each LanguagePlugin.

Migration path:
1. Use LanguageRegistry.get(language).tree_sitter_queries for new code
2. These dicts remain for backward compatibility during transition
3. Eventually remove once all callers migrate to plugin-based approach
"""

from typing import Dict, List

# =============================================================================
# PRIMITIVE / CONTAINER TYPES (exclude from COMPOSED_OF phantom nodes)
# =============================================================================
_PRIMITIVE_TYPES = frozenset(
    {
        # Rust
        "String",
        "str",
        "Vec",
        "Option",
        "Result",
        "Box",
        "Arc",
        "Rc",
        "HashMap",
        "HashSet",
        "BTreeMap",
        "BTreeSet",
        "VecDeque",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "f32",
        "f64",
        "bool",
        "char",
        "PathBuf",
        "Duration",
        "Instant",
        "Cow",
        # JS/TS
        "string",
        "number",
        "boolean",
        "any",
        "void",
        "null",
        "undefined",
        "Array",
        "Map",
        "Set",
        "Promise",
        "Date",
        "RegExp",
        "Error",
        "Record",
        "Partial",
        "Required",
        "Readonly",
        # Java/C#
        "int",
        "long",
        "float",
        "double",
        "byte",
        "short",
        "Integer",
        "Long",
        "Float",
        "Double",
        "Boolean",
        "List",
        "ArrayList",
        "LinkedList",
        "Object",
    }
)


# Map file extensions to tree-sitter language ids
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".json": "config-json",
    ".yaml": "config-yaml",
    ".yml": "config-yaml",
    ".toml": "config-toml",
    ".ini": "config-ini",
    ".properties": "config-properties",
    ".conf": "config-hocon",
    ".hocon": "config-hocon",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
}


REFERENCE_QUERIES: Dict[str, str] = {
    "python": """
        (call function: (identifier) @name)
        (call function: (attribute attribute: (identifier) @name))
        (attribute object: (_) attribute: (identifier) @name)
        (identifier) @name
    """,
    "javascript": """
        (call_expression function: (identifier) @name)
        (call_expression function: (member_expression property: (property_identifier) @name))
        (member_expression property: (property_identifier) @name)
        (new_expression constructor: (identifier) @name)
        (identifier) @name
    """,
    "typescript": """
        (call_expression function: (identifier) @name)
        (call_expression function: (member_expression property: (property_identifier) @name))
        (member_expression property: (property_identifier) @name)
        (new_expression constructor: (identifier) @name)
        (identifier) @name
    """,
    "java": """
        (method_invocation name: (identifier) @name)
        (method_invocation object: (identifier) @name)
        (field_access field: (identifier) @name)
    """,
    "go": """
        (call_expression function: (identifier) @name)
        (call_expression function: (selector_expression field: (field_identifier) @name))
        (selector_expression field: (field_identifier) @name)
        (identifier) @name
    """,
}


# Tree-sitter symbol queries per language for lightweight multi-language graph capture.
SYMBOL_QUERIES: Dict[str, List[tuple[str, str]]] = {
    "python": [
        ("class", "(class_definition name: (identifier) @name) @def"),
        ("function", "(function_definition name: (identifier) @name) @def"),
    ],
    "javascript": [
        ("class", "(class_declaration name: (identifier) @name) @def"),
        ("function", "(function_declaration name: (identifier) @name) @def"),
        ("function", "(method_definition name: (property_identifier) @name) @def"),
        (
            "function",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function))) @def",
        ),
        (
            "function",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression))) @def",
        ),
        (
            "function",
            "(assignment_expression left: (identifier) @name right: (arrow_function)) @def",
        ),
    ],
    "typescript": [
        ("class", "(class_declaration name: (identifier) @name) @def"),
        ("function", "(function_declaration name: (identifier) @name) @def"),
        ("function", "(method_signature name: (property_identifier) @name) @def"),
        ("function", "(method_definition name: (property_identifier) @name) @def"),
        (
            "function",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function))) @def",
        ),
        (
            "function",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression))) @def",
        ),
        (
            "function",
            "(assignment_expression left: (identifier) @name right: (arrow_function)) @def",
        ),
    ],
    "go": [
        ("function", "(function_declaration name: (identifier) @name) @def"),
        ("function", "(method_declaration name: (field_identifier) @name) @def"),
        ("class", "(type_declaration (type_spec name: (type_identifier) @name)) @def"),
    ],
    "java": [
        ("class", "(class_declaration name: (identifier) @name) @def"),
        ("class", "(interface_declaration name: (identifier) @name) @def"),
        ("function", "(method_declaration name: (identifier) @name) @def"),
    ],
    "cpp": [
        ("class", "(class_specifier name: (type_identifier) @name) @def"),
        (
            "function",
            "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def",
        ),
        (
            "function",
            "(function_definition declarator: (function_declarator declarator: (field_identifier) @name)) @def",
        ),
    ],
}

INHERITS_QUERIES: Dict[str, str] = {
    "python": """
        (class_definition
            name: (identifier) @child
            superclasses: (argument_list (identifier) @base))
    """,
    "javascript": """
        (class_declaration
            name: (identifier) @child
            (class_heritage (identifier) @base))
    """,
    "typescript": """
        (class_declaration
            name: (identifier) @child
            (class_heritage (identifier) @base))
    """,
    "java": """
        (class_declaration
            name: (identifier) @child
            super_classes: (superclass (type_identifier) @base))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @child
            (base_class_clause (base_class (type_identifier) @base))
        )
    """,
}

IMPLEMENTS_QUERIES: Dict[str, str] = {
    "typescript": """
        (class_declaration
            name: (type_identifier) @child
            (class_heritage
                (implements_clause (type_identifier) @interface)))
    """,
    "java": """
        (class_declaration
            name: (identifier) @child
            interfaces: (super_interfaces (type_list (type_identifier) @interface)))
        (interface_declaration
            name: (identifier) @child
            interfaces: (super_interfaces (type_list (type_identifier) @interface)))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @child
            (base_class_clause (base_class (type_identifier) @base))
        )
    """,
}

COMPOSITION_QUERIES: Dict[str, str] = {
    "javascript": """
        (class_declaration
            name: (identifier) @owner
            body: (class_body
                (method_definition
                    body: (statement_block
                        (expression_statement
                            (assignment_expression
                                left: (member_expression object: (this) property: (property_identifier))
                                right: (new_expression constructor: (identifier) @type)))))))
    """,
    "typescript": """
        (class_declaration
            name: (identifier) @owner
            body: (class_body
                (field_definition
                    type: (type_annotation (type_identifier) @type))
                (public_field_definition
                    type: (type_annotation (type_identifier) @type))
                (method_definition
                    body: (statement_block
                        (expression_statement
                            (assignment_expression
                                left: (member_expression object: (this) property: (property_identifier))
                                right: (new_expression constructor: (identifier) @type)))))))
    """,
    "go": """
        (type_declaration
            (type_spec
                name: (type_identifier) @owner
                type: (struct_type
                    (field_declaration
                        type: (type_identifier) @type))))
    """,
    "java": """
        (class_declaration
            name: (identifier) @owner
            body: (class_body
                (field_declaration
                    type: (type_identifier) @type)))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @owner
            body: (field_declaration_list
                (field_declaration
                    type: (type_identifier) @type)))
    """,
}

# Tree-sitter call queries (callee only) for multi-language call/reference edges.
CALL_QUERIES: Dict[str, str] = {
    "python": """
        (call function: (identifier) @callee)
        (call function: (attribute attribute: (identifier) @callee))
    """,
    "javascript": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (member_expression property: (property_identifier) @callee))
        (call_expression function: (subscript_expression index: (property_identifier) @callee))
        (new_expression constructor: (identifier) @callee)
    """,
    "typescript": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (member_expression property: (property_identifier) @callee))
        (call_expression function: (subscript_expression index: (property_identifier) @callee))
        (new_expression constructor: (identifier) @callee)
    """,
    "go": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (selector_expression field: (field_identifier) @callee))
        (type_conversion_expression type: (type_identifier) @callee)
    """,
    "java": """
        (method_invocation name: (identifier) @callee)
        (object_creation_expression type: (type_identifier) @callee)
        (super_method_invocation name: (identifier) @callee)
    """,
    "cpp": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (field_expression field: (field_identifier) @callee))
        (new_expression type: (type_identifier) @callee)
    """,
}

# Mapping of function/method node types to name field for caller resolution.
ENCLOSING_NAME_FIELDS: Dict[str, List[tuple[str, str]]] = {
    "python": [
        ("function_definition", "name"),
        ("class_definition", "name"),
    ],
    "javascript": [
        ("function_declaration", "name"),
        ("method_definition", "name"),
        ("class_declaration", "name"),
    ],
    "typescript": [
        ("function_declaration", "name"),
        ("method_definition", "name"),
        ("method_signature", "name"),
        ("class_declaration", "name"),
    ],
    "go": [
        ("function_declaration", "name"),
        ("method_declaration", "name"),
    ],
    "java": [
        ("method_declaration", "name"),
        ("class_declaration", "name"),
        ("interface_declaration", "name"),
    ],
    "cpp": [
        ("function_definition", "declarator"),
        ("class_specifier", "name"),
    ],
}
