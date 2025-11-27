"""
Local Code Executor for Excel Analysis
Executes Python code locally with access to Excel files in ./excel_files/
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from io import StringIO
from typing import Dict, Any, Optional
import traceback
import re


class ExcelCodeExecutor:
    """
    Executes Python code locally to analyze Excel files.
    Provides safe sandbox for pandas operations.
    """

    def __init__(self, excel_files_dir: str = "./excel_files"):
        self.excel_files_dir = excel_files_dir

    def execute_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """
        Execute Python code locally with access to the Excel file.

        Args:
            code: Python code to execute (should use 'df' variable)
            file_path: Path to Excel/CSV file

        Returns:
            Dict with 'success', 'result', 'output', 'error'
        """
        # Validate file exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "result": None,
                "output": ""
            }

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Load the data
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_path}",
                    "result": None,
                    "output": ""
                }

            # Create safe execution environment
            exec_globals = {
                'pd': pd,
                'np': np,
                'df': df,
                'json': json,
                'print': print,
            }
            exec_locals = {}

            # Execute the code
            exec(code, exec_globals, exec_locals)

            # Get the result (look for 'result' variable)
            result = exec_locals.get('result', None)

            # If no result variable, try to get the last expression value
            if result is None and exec_locals:
                # Get the last defined variable that's not a function/module
                for key, value in reversed(list(exec_locals.items())):
                    if not callable(value) and not key.startswith('_'):
                        result = value
                        break

            # Restore stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()

            # Convert result to serializable format
            result_serialized = self._serialize_result(result)

            return {
                "success": True,
                "result": result_serialized,
                "output": output,
                "error": None
            }

        except Exception as e:
            sys.stdout = old_stdout
            output = captured_output.getvalue()

            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc(),
                "result": None,
                "output": output
            }

    def _serialize_result(self, result: Any) -> Any:
        """Convert result to JSON-serializable format"""
        if result is None:
            return None
        elif isinstance(result, (int, float, str, bool)):
            return result
        elif isinstance(result, (np.integer, np.floating)):
            return result.item()
        elif isinstance(result, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": result.shape,
                "data": result.head(100).to_dict('records'),  # Limit to 100 rows
                "columns": result.columns.tolist()
            }
        elif isinstance(result, pd.Series):
            return {
                "type": "Series",
                "data": result.head(100).to_dict(),  # Limit to 100 items
                "name": result.name
            }
        elif isinstance(result, (list, tuple)):
            return [self._serialize_result(item) for item in result[:100]]  # Limit to 100 items
        elif isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in list(result.items())[:100]}
        else:
            return str(result)

    def get_file_metadata(self, file_path: str, search_keywords: list = None) -> Dict[str, Any]:
        """
        Get metadata about Excel/CSV file without loading full data.
        Optionally searches for relevant rows based on keywords.

        Args:
            file_path: Path to Excel/CSV file
            search_keywords: Optional list of keywords to search for relevant rows

        Returns:
            Dict with row_count, columns, sample_data, dtypes
        """
        try:
            # Load full data for searching
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df_full = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df_full = pd.read_csv(file_path)
            else:
                return {"error": f"Unsupported file type: {file_path}"}

            row_count = len(df_full)

            # Get sample data - either search-based or first rows
            if search_keywords and len(search_keywords) > 0:
                # Search for relevant rows containing keywords
                sample_data = self._get_relevant_sample(df_full, search_keywords)
            else:
                # Default: first 10 rows
                sample_data = df_full.head(10).to_dict('records')

            metadata = {
                "row_count": row_count,
                "column_count": len(df_full.columns),
                "columns": df_full.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df_full.dtypes.items()},
                "sample_data": sample_data,
                "numeric_columns": df_full.select_dtypes(include=[np.number]).columns.tolist(),
                "sample_note": f"Showing relevant rows matching keywords: {', '.join(search_keywords)}" if search_keywords else "Showing first 10 rows"
            }

            return metadata

        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}

    def _get_relevant_sample(self, df: pd.DataFrame, keywords: list, max_rows: int = 10) -> list:
        """
        Search DataFrame for rows containing keywords and return relevant samples.

        Args:
            df: DataFrame to search
            keywords: List of keywords to search for
            max_rows: Maximum number of sample rows to return

        Returns:
            List of dicts representing relevant rows
        """
        relevant_rows = []

        # Convert all columns to string for searching
        df_str = df.astype(str)

        # Search for rows containing any keyword
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Search across all columns
            mask = df_str.apply(lambda row: row.str.contains(keyword_lower, case=False, na=False).any(), axis=1)
            matching_rows = df[mask]

            if len(matching_rows) > 0:
                # Add up to 5 matching rows per keyword
                relevant_rows.extend(matching_rows.head(5).to_dict('records'))

                # Stop if we have enough samples
                if len(relevant_rows) >= max_rows:
                    break

        # Remove duplicates while preserving order
        seen = set()
        unique_rows = []
        for row in relevant_rows:
            row_tuple = tuple(sorted(row.items()))
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_rows.append(row)

        # If no relevant rows found, return first rows
        if len(unique_rows) == 0:
            unique_rows = df.head(max_rows).to_dict('records')

        # Limit to max_rows
        return unique_rows[:max_rows]


def extract_python_code(text: str) -> Optional[str]:
    """
    Extract Python code from AI response.
    Looks for code blocks marked with ```python or ```
    """
    # Try to find code block with language specifier
    pattern1 = r"```python\n(.*?)\n```"
    match = re.search(pattern1, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block
    pattern2 = r"```\n(.*?)\n```"
    match = re.search(pattern2, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code blocks, return None
    return None


if __name__ == "__main__":
    # Test the executor
    executor = ExcelCodeExecutor()

    # Example test
    test_code = """
# Calculate sum of Amount column
result = df['Amount'].sum()
print(f"Total amount: {result}")
"""

    result = executor.execute_code(test_code, "./excel_files/test.xlsx")
    print(json.dumps(result, indent=2))
