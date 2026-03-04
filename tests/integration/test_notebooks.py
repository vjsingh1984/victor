#!/usr/bin/env python
"""Test Jupyter notebooks by executing all cells.

This script runs through each tutorial notebook and executes all cells,
reporting which notebooks pass/fail and which specific cells had errors.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nbformat
from nbformat.v4 import new_notebook
from jupyter_client import KernelManager


class NotebookTester:
    """Test executor for Jupyter notebooks."""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    async def execute_notebook(
        self, notebook_path: Path
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Execute a notebook and return results.

        Args:
            notebook_path: Path to the notebook file

        Returns:
            Tuple of (success, cell_results)
        """
        # Load the notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        kernel_name = nb.metadata.get("kernelspec", {}).get("name", "python3")

        # Start kernel
        km = KernelManager(kernel_name=kernel_name)
        km.start_kernel()

        kc = km.client()
        kc.wait_for_ready(timeout=30)

        cell_results = []
        all_passed = True

        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            result = {
                "cell_num": i,
                "source": cell.source[:100] + "..." if len(cell.source) > 100 else cell.source,
                "execution_count": cell.execution_count,
                "outputs": [],
                "error": None,
            }

            try:
                # Execute the cell
                msg_id = kc.execute(cell.source)

                # Wait for execution to complete
                while True:
                    msg = kc.get_iopub_msg(timeout=30)
                    if msg["content"] == "idle" or msg["content"] == "complete":
                        break

                    if msg["msg_type"] == "stream":
                        result["outputs"].append({
                            "type": "stream",
                            "name": msg["content"].get("name", ""),
                            "text": msg["content"].get("text", ""),
                        })
                    elif msg["msg_type"] == "execute_result":
                        result["outputs"].append({
                            "type": "execute_result",
                            "data": msg["content"].get("data", {}),
                            "metadata": msg["content"].get("metadata", {}),
                        })
                    elif msg["msg_type"] == "error":
                        result["error"] = {
                            "ename": msg["content"].get("ename", ""),
                            "evalue": msg["content"].get("evalue", ""),
                            "traceback": msg["content"].get("traceback", []),
                        }
                        all_passed = False

            except Exception as e:
                result["error"] = {
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [],
                }
                all_passed = False

            cell_results.append(result)

        # Shutdown kernel
        kc.stop_channels()
        km.shutdown_kernel()

        return all_passed, cell_results

    async def test_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """Test a single notebook.

        Args:
            notebook_path: Path to the notebook

        Returns:
            Test result dictionary
        """
        print(f"\n{'='*60}")
        print(f"Testing: {notebook_path.name}")
        print(f"{'='*60}")

        try:
            success, cell_results = await self.execute_notebook(notebook_path)

            failed_cells = [r for r in cell_results if r.get("error")]

            result = {
                "notebook": notebook_path.name,
                "success": success,
                "total_cells": len(cell_results),
                "failed_cells": len(failed_cells),
                "cells": cell_results,
            }

            # Print results
            if success:
                print(f"✅ PASSED - All {len(cell_results)} cells executed successfully")
            else:
                print(f"❌ FAILED - {len(failed_cells)}/{len(cell_results)} cells had errors")

                # Print details of failed cells
                for cell_result in failed_cells:
                    print(f"\n  Cell {cell_result['cell_num']}:")
                    print(f"    Source: {cell_result['source']}")
                    if cell_result.get("error"):
                        error = cell_result["error"]
                        print(f"    Error: {error['ename']}: {error['evalue']}")
                        if error.get("traceback"):
                            for line in error["traceback"][-5:]:  # Last 5 lines
                                print(f"        {line.strip()}")

            return result

        except Exception as e:
            print(f"❌ ERROR - Failed to execute notebook: {e}")
            return {
                "notebook": notebook_path.name,
                "success": False,
                "error": str(e),
                "total_cells": 0,
                "failed_cells": 0,
            }

    async def test_all_notebooks(self, notebooks_dir: Path) -> Dict[str, Any]:
        """Test all notebooks in a directory.

        Args:
            notebooks_dir: Directory containing notebooks

        Returns:
            Summary of all test results
        """
        notebook_files = sorted(notebooks_dir.glob("*.ipynb"))

        if not notebook_files:
            print(f"No notebooks found in {notebooks_dir}")
            return {"total": 0, "passed": 0, "failed": 0, "results": []}

        print(f"\nFound {len(notebook_files)} notebooks to test")

        results = []
        for notebook_file in notebook_files:
            result = await self.test_notebook(notebook_file)
            results.append(result)

        # Calculate summary
        passed = sum(1 for r in results if r.get("success", False))
        failed = len(results) - passed

        summary = {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "results": results,
        }

        # Print final summary
        print(f"\n{'='*60}")
        print("NOTEBOOK TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {summary['total']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")

        if failed > 0:
            print("\nFailed notebooks:")
            for r in results:
                if not r.get("success", False):
                    print(f"  - {r['notebook']}")

        return summary


async def main():
    """Main entry point."""
    notebooks_dir = Path("docs/tutorials/notebooks")

    if not notebooks_dir.exists():
        print(f"Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)

    tester = NotebookTester()
    summary = await tester.test_all_notebooks(notebooks_dir)

    # Exit with error code if any tests failed
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
