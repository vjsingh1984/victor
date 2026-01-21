#!/usr/bin/env python3
"""
Victor AI Documentation Generator - Web Interface

Flask web application for generating and viewing documentation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.agent.orchestrator_factory import create_orchestrator
from victor.config.settings import Settings
from victor.rag.tools import DocumentIndexer, SemanticSearch
from victor.workflows import YAMLWorkflowProvider


app = Flask(__name__)
app.config['SECRET_KEY'] = 'victor-doc-generator-demo'
app.config['UPLOAD_FOLDER'] = '/tmp/victor-docs'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Victor AI components
settings = Settings()
orchestrator = create_orchestrator(settings)
doc_indexer = DocumentIndexer()


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate_docs():
    """Generate documentation for uploaded project.

    Expects:
        - file: Project zip file
        - format: Output format (markdown, html, pdf)
        - options: Additional options (diagrams, examples, etc.)

    Returns:
        JSON response with generated documentation
    """
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get options
        output_format = request.form.get('format', 'markdown')
        include_diagrams = request.form.get('diagrams', 'true').lower() == 'true'
        include_examples = request.form.get('examples', 'true').lower() == 'true'

        # Generate documentation
        from src.doc_generator import DocumentationGenerator

        generator = DocumentationGenerator(orchestrator, doc_indexer)

        result = generator.generate(
            project_path=filepath,
            output_format=output_format,
            include_diagrams=include_diagrams,
            include_examples=include_examples
        )

        return jsonify({
            'success': True,
            'documentation': result,
            'stats': {
                'files_processed': result.get('file_count', 0),
                'functions_documented': result.get('function_count', 0),
                'classes_documented': result.get('class_count', 0),
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search_docs():
    """Search documentation using RAG.

    Expects:
        - query: Search query
        - index_path: Path to documentation index

    Returns:
        JSON response with search results
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        index_path = data.get('index_path', app.config['UPLOAD_FOLDER'])

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Perform semantic search
        search = SemanticSearch(doc_indexer)
        results = search.query(query, index_path=index_path)

        return jsonify({
            'success': True,
            'results': [
                {
                    'content': r['content'],
                    'score': r['score'],
                    'metadata': r.get('metadata', {})
                }
                for r in results
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def export_docs():
    """Export documentation to various formats.

    Expects:
        - documentation: Documentation content
        - format: Export format (markdown, html, pdf)
        - filename: Output filename

    Returns:
        File download
    """
    try:
        data = request.get_json()
        documentation = data.get('documentation', '')
        export_format = data.get('format', 'markdown')
        filename = data.get('filename', 'documentation')

        from src.exporter import DocumentationExporter

        exporter = DocumentationExporter()
        output_path = exporter.export(
            content=documentation,
            format=export_format,
            filename=filename,
            output_dir=app.config['UPLOAD_FOLDER']
        )

        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"{filename}.{export_format}"
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    """Analyze code structure and extract information.

    Expects:
        - code: Code snippet or file path

    Returns:
        JSON response with analysis results
    """
    try:
        data = request.get_json()
        code = data.get('code', '')

        if not code:
            return jsonify({'error': 'No code provided'}), 400

        from src.code_analyzer import CodeAnalyzer

        analyzer = CodeAnalyzer(orchestrator)
        result = analyzer.analyze(code)

        return jsonify({
            'success': True,
            'analysis': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
