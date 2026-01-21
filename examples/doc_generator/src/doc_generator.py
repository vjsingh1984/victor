"""
Documentation generator using Victor AI's research and RAG capabilities.
"""

import os
import zipfile
from pathlib import Path
from typing import Dict, Any, List

from victor.coding.ast import Parser
from victor.coding.codebase import CodebaseIndexer
from victor.rag.tools import DocumentIndexer
from victor.workflows import YAMLWorkflowProvider


class DocumentationGenerator:
    """Generate comprehensive documentation from code."""

    def __init__(self, orchestrator, doc_indexer: DocumentIndexer):
        """Initialize documentation generator.

        Args:
            orchestrator: Victor AI agent orchestrator
            doc_indexer: Document indexer for RAG
        """
        self.orchestrator = orchestrator
        self.doc_indexer = doc_indexer
        self.parser = Parser()
        self.codebase_indexer = CodebaseIndexer()

    def generate(self, project_path: str, output_format: str = "markdown",
                 include_diagrams: bool = True, include_examples: bool = True) -> Dict[str, Any]:
        """Generate documentation for a project.

        Args:
            project_path: Path to project (zip file or directory)
            output_format: Output format (markdown, html, pdf)
            include_diagrams: Include architecture diagrams
            include_examples: Include code examples

        Returns:
            Generated documentation
        """
        # Extract project if zip
        temp_dir = None
        if project_path.endswith('.zip'):
            temp_dir = self._extract_zip(project_path)
            project_path = temp_dir

        try:
            # Analyze codebase
            analysis = self._analyze_codebase(project_path)

            # Generate documentation sections
            sections = {
                'overview': self._generate_overview(analysis),
                'installation': self._generate_installation(analysis),
                'api': self._generate_api_docs(analysis),
                'architecture': self._generate_architecture_docs(analysis) if include_diagrams else None,
                'examples': self._generate_examples(analysis) if include_examples else None,
            }

            # Format output
            formatted_docs = self._format_output(sections, output_format)

            # Index for search
            self._index_documentation(formatted_docs, project_path)

            return {
                'content': formatted_docs,
                'format': output_format,
                'file_count': analysis['file_count'],
                'function_count': analysis['function_count'],
                'class_count': analysis['class_count'],
            }

        finally:
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

    def _extract_zip(self, zip_path: str) -> str:
        """Extract zip file to temporary directory."""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp(prefix='victor_docs_')

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        return temp_dir

    def _analyze_codebase(self, project_path: str) -> Dict[str, Any]:
        """Analyze codebase structure."""
        project_path = Path(project_path)

        # Find all code files
        code_files = []
        for ext in ['.py', '.js', '.ts', '.go', '.java']:
            code_files.extend(project_path.rglob(f'*{ext}'))

        # Parse each file
        all_functions = []
        all_classes = []
        all_modules = []

        for file_path in code_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse AST
                language = self._get_language(file_path)
                ast = self.parser.parse(content, language=language)

                # Extract information
                functions = self.codebase_indexer.extract_functions(ast)
                classes = self.codebase_indexer.extract_classes(ast)
                modules = self.codebase_indexer.extract_modules(ast)

                all_functions.extend(functions)
                all_classes.extend(classes)
                all_modules.append({
                    'path': str(file_path),
                    'name': file_path.stem,
                    'docstring': modules.get('docstring', ''),
                })

            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

        return {
            'file_count': len(code_files),
            'function_count': len(all_functions),
            'class_count': len(all_classes),
            'functions': all_functions,
            'classes': all_classes,
            'modules': all_modules,
        }

    def _generate_overview(self, analysis: Dict[str, Any]) -> str:
        """Generate project overview."""
        # Use Victor AI to generate intelligent overview
        prompt = f"""
        Generate a project overview based on this analysis:
        - Files: {analysis['file_count']}
        - Functions: {analysis['function_count']}
        - Classes: {analysis['class_count']}

        Key modules:
        {', '.join([m['name'] for m in analysis['modules'][:10]])}
        """

        overview = self.orchestrator.process_request(
            "Generate a concise project overview",
            context={"analysis": analysis}
        )

        return f"""
# Project Overview

{overview}

## Statistics

- **Total Files:** {analysis['file_count']}
- **Functions:** {analysis['function_count']}
- **Classes:** {analysis['class_count']}
"""

    def _generate_installation(self, analysis: Dict[str, Any]) -> str:
        """Generate installation instructions."""
        # Detect package managers
        has_requirements = any(
            Path(m['path']).parent / 'requirements.txt'
            for m in analysis['modules']
        )
        has_package_json = any(
            Path(m['path']).parent / 'package.json'
            for m in analysis['modules']
        )

        instructions = "# Installation\n\n"

        if has_requirements:
            instructions += """## Python Installation

```bash
pip install -r requirements.txt
```

"""

        if has_package_json:
            instructions += """## Node.js Installation

```bash
npm install
```

"""

        return instructions

    def _generate_api_docs(self, analysis: Dict[str, Any]) -> str:
        """Generate API documentation."""
        docs = "# API Documentation\n\n"

        # Document classes
        for cls in analysis['classes'][:20]:  # Limit to 20 classes
            docs += f"## {cls['name']}\n\n"
            if cls.get('docstring'):
                docs += f"{cls['docstring']}\n\n"

            # Document methods
            if 'methods' in cls:
                for method in cls['methods']:
                    docs += f"### {method['name']}\n\n"
                    if method.get('docstring'):
                        docs += f"{method['docstring']}\n\n"

                    if method.get('parameters'):
                        docs += "**Parameters:**\n"
                        for param in method['parameters']:
                            docs += f"- `{param['name']}` ({param['type']}): {param.get('description', '')}\n"
                        docs += "\n"

        # Document top-level functions
        docs += "\n## Functions\n\n"
        for func in analysis['functions'][:50]:  # Limit to 50 functions
            docs += f"### {func['name']}\n\n"
            if func.get('docstring'):
                docs += f"{func['docstring']}\n\n"

            if func.get('parameters'):
                docs += "**Parameters:**\n"
                for param in func['parameters']:
                    docs += f"- `{param['name']}` ({param['type']}): {param.get('description', '')}\n"
                docs += "\n"

        return docs

    def _generate_architecture_docs(self, analysis: Dict[str, Any]) -> str:
        """Generate architecture documentation with diagrams."""
        docs = "# Architecture\n\n"

        # Generate module dependency diagram
        docs += "## Module Dependencies\n\n"
        docs += "```mermaid\n"
        docs += "graph TD\n"

        for i, module in enumerate(analysis['modules'][:10]):
            docs += f"    {i}[{module['name']}]\n"
            if i > 0:
                docs += f"    {i-1} --> {i}\n"

        docs += "```\n\n"

        # Generate class hierarchy diagram
        if analysis['classes']:
            docs += "## Class Hierarchy\n\n"
            docs += "```mermaid\n"
            docs += "graph TB\n"

            for i, cls in enumerate(analysis['classes'][:10]):
                docs.append(f"    {i}[{cls['name']}]\n")
                if 'base_class' in cls and cls['base_class']:
                    docs += f"    {i} -.-> {cls['base_class']}\n"

            docs += "```\n\n"

        return docs

    def _generate_examples(self, analysis: Dict[str, Any]) -> str:
        """Generate usage examples."""
        # Extract examples from docstrings
        examples = []

        for func in analysis['functions']:
            if func.get('docstring') and 'Example' in func['docstring']:
                examples.append({
                    'name': func['name'],
                    'example': func['docstring']
                })

        docs = "# Usage Examples\n\n"

        for example in examples[:10]:
            docs += f"## {example['name']}\n\n"
            docs += f"{example['example']}\n\n"

        return docs

    def _format_output(self, sections: Dict[str, str], output_format: str) -> str:
        """Format documentation output."""
        # Combine sections
        combined = []
        for section_name, content in sections.items():
            if content:
                combined.append(content)

        documentation = "\n\n".join(combined)

        # Convert to requested format
        if output_format == "markdown":
            return documentation
        elif output_format == "html":
            return self._convert_to_html(documentation)
        elif output_format == "pdf":
            # Return markdown, will be converted by exporter
            return documentation
        else:
            return documentation

    def _convert_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML."""
        import markdown

        html = markdown.markdown(markdown)

        # Wrap in HTML template
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>API Documentation</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
{html}
</body>
</html>
        """

    def _index_documentation(self, documentation: str, project_path: str):
        """Index documentation for RAG search."""
        # Split into chunks
        chunks = self._chunk_documentation(documentation)

        # Index chunks
        for i, chunk in enumerate(chunks):
            self.doc_indexer.add_document(
                doc_id=f"{project_path}_{i}",
                content=chunk,
                metadata={"source": project_path}
            )

    def _chunk_documentation(self, documentation: str, chunk_size: int = 1000) -> List[str]:
        """Split documentation into chunks for indexing."""
        chunks = []
        lines = documentation.split('\n')
        current_chunk = []
        current_size = 0

        for line in lines:
            if current_size + len(line) > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += len(line)

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _get_language(self, file_path: Path) -> str:
        """Determine programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".java": "java",
        }
        return ext_map.get(file_path.suffix, "python")
