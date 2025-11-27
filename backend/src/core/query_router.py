"""
Intelligent Query Router
Determines the best strategy to answer user queries
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class QueryPlan:
    """Plan for executing a query"""
    query_type: str  # 'summary', 'analysis', 'question', 'calculation'
    target_files: List[str]  # Which files to use
    requires_code: bool  # Whether code execution is needed
    strategy: str  # 'retrieve', 'code_excel', 'multi_file', 'summary'
    reasoning: str  # Why this plan was chosen


class IntelligentQueryRouter:
    """Routes queries to the appropriate handler based on intent and context"""

    def __init__(self, ai_model=None):
        self.ai_model = ai_model  # AI model for intelligent classification

        # Query type indicators (fallback if AI not available)
        self.summary_indicators = [
            'summarize', 'summary', 'overview', 'what is this about',
            'tell me about', 'describe', 'main points', 'key points',
            'brief', 'explain the document', 'what does it contain'
        ]

        self.calculation_indicators = [
            'total', 'sum', 'count', 'average', 'mean', 'median',
            'how many', 'calculate', 'aggregate', 'percentage',
            'spent', 'spend', 'earned', 'earn', 'more', 'less',
            'amount', 'balance', 'difference', 'compare amounts'
        ]

        self.analysis_indicators = [
            'filter', 'find all', 'list', 'show', 'transactions',
            'where', 'between', 'from', 'to', 'containing',
            'search for', 'look for'
        ]

    def plan_query(self, question: str, available_files: List[Dict[str, Any]]) -> QueryPlan:
        """
        Create an execution plan for the query.

        Args:
            question: User's question
            available_files: List of available files with metadata
                [{'filename': 'x.xlsx', 'type': 'excel', 'row_count': 100}, ...]

        Returns:
            QueryPlan with execution strategy
        """
        question_lower = question.lower()

        # Categorize available files
        excel_files = [f for f in available_files if f.get('type') == 'excel']
        pdf_files = [f for f in available_files if f.get('type') == 'pdf']
        other_files = [f for f in available_files if f.get('type') not in ['excel', 'pdf']]

        # Check if this is a multi-file scenario
        has_multiple_file_types = len([ft for ft in [excel_files, pdf_files, other_files] if ft]) > 1

        # Use AI classification if available
        if self.ai_model and excel_files:
            try:
                ai_classification = self._classify_with_ai(question, excel_files, has_multiple_file_types)
                if ai_classification:
                    # Log AI decision (will be picked up by logger if imported)
                    try:
                        from src.utils.logger_config import rag_logger
                        rag_logger.info(f"🤖 AI Classification: {ai_classification}")
                    except:
                        pass

                    if ai_classification == 'calculation':
                        return self._plan_calculation(question, excel_files)
                    elif ai_classification == 'analysis':
                        # If asking for "information" and we have multiple file types, use retrieval
                        if has_multiple_file_types and self._is_information_request(question_lower):
                            return self._plan_general_question(question, available_files)
                        return self._plan_analysis(question, excel_files)
                    elif ai_classification == 'summary':
                        return self._plan_summary(question, available_files, excel_files, pdf_files)
                    elif ai_classification == 'general':
                        return self._plan_general_question(question, available_files)
            except Exception as e:
                # AI classification failed, use fallback
                try:
                    from src.utils.logger_config import rag_logger
                    rag_logger.warning(f"⚠️ AI classification failed, using keywords: {e}")
                except:
                    pass

        # Fallback to keyword-based classification
        # 1. SUMMARY requests
        if self._is_summary_request(question_lower):
            return self._plan_summary(question, available_files, excel_files, pdf_files)

        # 2. INFORMATION requests across multiple files
        if has_multiple_file_types and self._is_information_request(question_lower):
            return self._plan_general_question(question, available_files)

        # 3. CALCULATION/ANALYSIS on structured data
        if self._is_calculation_request(question_lower) and excel_files:
            return self._plan_calculation(question, excel_files)

        # 4. DATA ANALYSIS (filtering, finding) - only if no other file types
        if self._is_analysis_request(question_lower) and excel_files and not has_multiple_file_types:
            return self._plan_analysis(question, excel_files)

        # 5. GENERAL QUESTION (retrieve from documents)
        return self._plan_general_question(question, available_files)

    def _classify_with_ai(self, question: str, excel_files: List[Dict],
                         has_multiple_file_types: bool = False) -> Optional[str]:
        """
        Use AI to classify query type intelligently.

        Returns: 'calculation', 'analysis', 'summary', or 'general'
        """
        if not self.ai_model:
            return None

        file_info = "\n".join([f"- {f['filename']} ({f.get('row_count', '?')} rows)" for f in excel_files[:3]])

        multi_file_note = ""
        if has_multiple_file_types:
            multi_file_note = """
**IMPORTANT:** User has MULTIPLE file types (Excel + PDF/documents).
- If asking for "information", "details", "any data on..." → answer: general (search all files)
- Only use "analysis" if specifically asking to analyze Excel data
"""

        prompt = f"""Classify this user query for a data analysis system.

**Available Excel Files:**
{file_info}
{multi_file_note}
**User Question:** "{question}"

**Classification Options:**
1. **calculation** - User wants to calculate, count, sum, compare amounts, or perform math operations
   Examples: "total spent?", "how many rows?", "spent vs earned?", "average amount?"

2. **analysis** - User wants to filter/search Excel data specifically
   Examples: "show all CREDIT entries from Excel", "list transactions from March in the spreadsheet"

3. **summary** - User wants overview or description of files
   Examples: "what's in this file?", "summarize the data", "describe the columns"

4. **general** - User wants information that could be in any document (Excel, PDF, etc.)
   Examples: "any information on X?", "find details about Y", "what does it say about Z?"

**Instructions:**
- If question involves math/totals/calculations → answer: calculation
- If question specifically asks to analyze/filter Excel → answer: analysis
- If question asks for overview/summary → answer: summary
- If asking for "information", "details", "any data on..." → answer: general
- Otherwise → answer: general

Answer with ONLY ONE WORD: calculation, analysis, summary, or general"""

        try:
            response = self.ai_model.generate_content(prompt)
            classification = response.text.strip().lower()

            # Extract first word if AI gave explanation
            if ' ' in classification:
                classification = classification.split()[0]

            if classification in ['calculation', 'analysis', 'summary', 'general']:
                return classification

            return None
        except:
            return None

    def _is_summary_request(self, question_lower: str) -> bool:
        """Check if query is asking for summary"""
        return any(indicator in question_lower for indicator in self.summary_indicators)

    def _is_calculation_request(self, question_lower: str) -> bool:
        """Check if query requires calculation"""
        return any(indicator in question_lower for indicator in self.calculation_indicators)

    def _is_analysis_request(self, question_lower: str) -> bool:
        """Check if query requires data analysis"""
        return any(indicator in question_lower for indicator in self.analysis_indicators)

    def _is_information_request(self, question_lower: str) -> bool:
        """Check if query is asking for general information (multi-file search)"""
        information_indicators = [
            'information', 'info', 'details', 'detail', 'any data',
            'anything', 'tell me about', 'what about', 'know about',
            'any', 'find', 'search', 'look for'
        ]
        return any(indicator in question_lower for indicator in information_indicators)

    def _plan_summary(self, question: str, all_files: List[Dict],
                     excel_files: List[Dict], pdf_files: List[Dict]) -> QueryPlan:
        """Plan for summary requests"""

        # If asking for summary of all files
        if 'all' in question.lower() or len(all_files) > 1:
            return QueryPlan(
                query_type='summary',
                target_files=[f['filename'] for f in all_files],
                requires_code=False,
                strategy='multi_file_summary',
                reasoning='User wants summary of multiple documents - retrieve and synthesize content'
            )

        # Summary of single file
        return QueryPlan(
            query_type='summary',
            target_files=[all_files[0]['filename']] if all_files else [],
            requires_code=False,
            strategy='summary',
            reasoning='User wants document summary - retrieve main content and summarize'
        )

    def _plan_calculation(self, question: str, excel_files: List[Dict]) -> QueryPlan:
        """Plan for calculation requests"""

        # Extract file references from question
        target_files = self._identify_target_files(question, excel_files)

        if not target_files:
            # Default to first Excel file
            target_files = [excel_files[0]['filename']]

        return QueryPlan(
            query_type='calculation',
            target_files=target_files,
            requires_code=True,
            strategy='code_excel',
            reasoning='User needs calculation from Excel data - use code execution for accuracy'
        )

    def _plan_analysis(self, question: str, excel_files: List[Dict]) -> QueryPlan:
        """Plan for data analysis requests"""

        target_files = self._identify_target_files(question, excel_files)

        if not target_files:
            target_files = [excel_files[0]['filename']]

        return QueryPlan(
            query_type='analysis',
            target_files=target_files,
            requires_code=True,
            strategy='code_excel',
            reasoning='User needs data filtering/search - use code execution for precision'
        )

    def _plan_general_question(self, question: str, all_files: List[Dict]) -> QueryPlan:
        """Plan for general questions"""

        # Try to identify relevant files from question
        target_files = self._identify_target_files(question, all_files)

        if not target_files and all_files:
            # Search across all files
            target_files = [f['filename'] for f in all_files]

        return QueryPlan(
            query_type='question',
            target_files=target_files,
            requires_code=False,
            strategy='retrieve',
            reasoning='General question - retrieve relevant context from documents'
        )

    def _identify_target_files(self, question: str, files: List[Dict]) -> List[str]:
        """
        Identify which files are mentioned or relevant to the question.

        Args:
            question: User's question
            files: List of available files

        Returns:
            List of filenames that are relevant
        """
        target_files = []
        question_lower = question.lower()

        for file_info in files:
            filename = file_info['filename']
            filename_lower = filename.lower()

            # Remove extension for matching
            name_without_ext = filename_lower.rsplit('.', 1)[0]

            # Check if filename is mentioned
            if name_without_ext in question_lower:
                target_files.append(filename)
                continue

            # Check for partial matches (e.g., "bank" matches "bank_statement.xlsx")
            name_parts = re.findall(r'\w+', name_without_ext)
            if any(part in question_lower for part in name_parts if len(part) > 3):
                target_files.append(filename)

        return target_files

    def extract_keywords(self, question: str) -> List[str]:
        """Extract meaningful keywords from question for searching"""
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'many',
            'total', 'sum', 'count', 'show', 'list', 'find', 'all', 'get',
            'calculate', 'tell', 'me', 'of', 'in', 'on', 'at', 'to', 'from',
            'for', 'with', 'by', 'about', 'that', 'this', 'these', 'those',
            'do', 'does', 'did', 'can', 'could', 'should', 'would', 'will',
            'there', 'transactions', 'entries', 'records', 'rows', 'data',
            'and', 'or', 'but', 'not', 'if', 'then', 'else', 'when', 'where'
        }

        # Extract words
        words = re.findall(r'\b\w+\b', question.lower())

        # Filter and return
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:5]  # Top 5 keywords
