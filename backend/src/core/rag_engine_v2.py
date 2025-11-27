"""
RAG Engine V2 - Production Ready
Intelligent multi-file query system with smart routing
"""

import os
import time
from typing import List, Dict, Tuple, Optional, Any
import google.generativeai as genai
import chromadb
from chromadb.config import Settings

from src.utils.logger_config import rag_logger
from src.executors.excel_code_executor import ExcelCodeExecutor, extract_python_code
from src.core.query_router import IntelligentQueryRouter, QueryPlan


class RAGEngineV2:
    """
    Production-ready RAG engine with intelligent query routing.
    Handles multiple files, smart context retrieval, and code execution.
    """

    def __init__(self, api_key: str, collection_name: str = "documents",
                 persist_directory: str = "./chroma_db"):
        """Initialize RAG Engine"""
        genai.configure(api_key=api_key)

        # Models
        self.embedding_model = 'text-embedding-004'
        self.generation_model = 'gemini-2.0-flash-exp'
        self.model = genai.GenerativeModel(self.generation_model)

        # Vector database
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Components
        self.query_router = IntelligentQueryRouter(ai_model=self.model)
        self.excel_executor = ExcelCodeExecutor()

        rag_logger.info("✅ RAG Engine V2 initialized")

    # ===== EMBEDDING =====

    def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embedding for text"""
        try:
            result = genai.embed_content(
                model=f"models/{self.embedding_model}",
                content=text,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            rag_logger.error(f"Embedding error: {e}")
            raise

    # ===== DOCUMENT MANAGEMENT =====

    def add_documents(self, chunks: List[str], metadatas: List[dict]) -> int:
        """Add document chunks to vector store"""
        if not chunks:
            return 0

        embeddings = []
        ids = []

        for i, chunk in enumerate(chunks):
            embedding = self.generate_embedding(chunk)
            embeddings.append(embedding)
            ids.append(f"{metadatas[i]['filename']}_{metadatas[i].get('chunk_number', i)}")

        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

        rag_logger.info(f"💾 Added {len(chunks)} chunks to database")
        return len(chunks)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with metadata"""
        try:
            results = self.collection.get()

            if not results['metadatas']:
                return []

            # Group by filename
            files_dict = {}
            for metadata in results['metadatas']:
                filename = metadata['filename']
                if filename not in files_dict:
                    files_dict[filename] = {
                        'filename': filename,
                        'upload_date': metadata.get('upload_date', 'Unknown'),
                        'content_type': metadata.get('content_type', 'document'),
                        'chunks': 0,
                        'total_words': 0
                    }

                    # Add Excel-specific info
                    if metadata.get('content_type') == 'excel_metadata':
                        files_dict[filename]['row_count'] = metadata.get('row_count', 0)
                        files_dict[filename]['col_count'] = metadata.get('col_count', 0)

                files_dict[filename]['chunks'] += 1
                # Estimate words per chunk (rough average)
                files_dict[filename]['total_words'] += 400

            return list(files_dict.values())

        except Exception as e:
            rag_logger.error(f"Error listing documents: {e}")
            return []

    def delete_document(self, filename: str) -> int:
        """Delete document and all its chunks"""
        try:
            results = self.collection.get(where={"filename": filename})

            if not results['ids']:
                return 0

            # Delete physical Excel files
            for metadata in results['metadatas']:
                if metadata.get('content_type') == 'excel_metadata':
                    file_path = metadata.get('file_path')
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                        rag_logger.info(f"🗑️ Deleted file: {file_path}")

            # Delete from database
            self.collection.delete(ids=results['ids'])
            rag_logger.info(f"🗑️ Deleted {len(results['ids'])} chunks")

            return len(results['ids'])

        except Exception as e:
            rag_logger.error(f"Delete error: {e}")
            raise

    # ===== CORE QUERY =====

    def query(self, question: str) -> Dict[str, Any]:
        """
        Main query entry point with intelligent routing.

        This method:
        1. Analyzes the question
        2. Lists available files
        3. Creates an execution plan
        4. Executes the plan
        5. Returns formatted results
        """
        start_time = time.time()
        rag_logger.log_query_start(question)

        try:
            # Get available files
            available_files = self._get_available_files()

            if not available_files:
                return {
                    'answer': "Hey! It looks like you haven't uploaded any documents yet. Upload some files and I'll be happy to help you analyze them!",
                    'sources': [],
                    'has_context': False
                }

            # Create query plan
            plan = self.query_router.plan_query(question, available_files)

            rag_logger.info(f"📋 Query Plan:")
            rag_logger.info(f"   Type: {plan.query_type}")
            rag_logger.info(f"   Strategy: {plan.strategy}")
            rag_logger.info(f"   Target files: {plan.target_files}")
            rag_logger.info(f"   Requires code: {plan.requires_code}")
            rag_logger.info(f"   Reasoning: {plan.reasoning}")

            # Execute based on strategy
            if plan.strategy == 'code_excel':
                result = self._execute_excel_code(question, plan)
            elif plan.strategy == 'multi_file_summary':
                result = self._execute_multi_file_summary(question, plan)
            elif plan.strategy == 'summary':
                result = self._execute_summary(question, plan)
            else:  # 'retrieve'
                result = self._execute_retrieval(question, plan)

            # Add metadata
            result['query_type'] = plan.query_type
            result['strategy'] = plan.strategy
            result['processing_time_ms'] = (time.time() - start_time) * 1000

            rag_logger.log_query_complete(result['processing_time_ms'])

            return result

        except Exception as e:
            rag_logger.error(f"Query error: {e}")
            import traceback
            rag_logger.error(traceback.format_exc())
            raise

    # ===== EXECUTION STRATEGIES =====

    def _execute_excel_code(self, question: str, plan: QueryPlan) -> Dict[str, Any]:
        """Execute Excel analysis with code"""
        filename = plan.target_files[0]  # Primary file

        # Get file path from metadata
        results = self.collection.query(
            query_embeddings=[self.generate_embedding(question, "retrieval_query")],
            n_results=1,
            where={
                "$and": [
                    {"filename": filename},
                    {"content_type": "excel_metadata"}
                ]
            }
        )

        if not results['documents'] or not results['documents'][0]:
            return {
                'answer': f"Hmm, I couldn't find {filename}. Are you sure it was uploaded? You can check your uploaded files and try again!",
                'sources': [],
                'has_context': False
            }

        metadata = results['metadatas'][0][0]
        file_path = metadata['file_path']

        # Extract keywords for relevant sampling
        keywords = self.query_router.extract_keywords(question)
        rag_logger.info(f"🔍 Keywords: {keywords}")

        # Get metadata with relevant samples
        file_metadata = self.excel_executor.get_file_metadata(file_path, keywords)

        if 'error' in file_metadata:
            raise Exception(file_metadata['error'])

        rag_logger.info(f"📊 {filename}: {file_metadata['row_count']:,} rows × {file_metadata['column_count']} cols")

        # Generate code
        code = self._generate_analysis_code(question, filename, file_metadata)

        if not code:
            return {
                'answer': "Hmm, I'm having trouble figuring out how to analyze that. Could you try asking your question in a different way? I'm here to help!",
                'sources': [{'filename': filename}],
                'has_context': False
            }

        rag_logger.info("=" * 70)
        rag_logger.info("📝 GENERATED CODE:")
        rag_logger.info(code)
        rag_logger.info("=" * 70)

        # Execute locally
        exec_result = self.excel_executor.execute_code(code, file_path)

        rag_logger.info("=" * 70)
        rag_logger.info("⚡ EXECUTION RESULT:")
        rag_logger.info(f"Success: {exec_result['success']}")
        if exec_result['success']:
            rag_logger.info(f"Output: {exec_result['output']}")
            rag_logger.info(f"Result: {exec_result['result']}")
        else:
            rag_logger.info(f"Error: {exec_result['error']}")
        rag_logger.info("=" * 70)

        # Format answer
        structured_data = None
        if exec_result['success']:
            answer, structured_data = self._format_excel_answer(exec_result, code, filename, file_metadata)
        else:
            # Retry once with error feedback
            rag_logger.warning(f"⚠️ First attempt failed: {exec_result['error']}")
            code_fixed = self._fix_code(code, exec_result['error'], file_metadata)

            if code_fixed:
                exec_result = self.excel_executor.execute_code(code_fixed, file_path)
                if exec_result['success']:
                    answer, structured_data = self._format_excel_answer(exec_result, code_fixed, filename, file_metadata)
                else:
                    answer = self._format_error_answer(exec_result, filename)
            else:
                answer = self._format_error_answer(exec_result, filename)

        result = {
            'answer': answer,
            'sources': [{'filename': filename, 'upload_date': metadata.get('upload_date')}],
            'has_context': True,
            'code_executed': exec_result['success']
        }

        # Add structured data if available
        if structured_data:
            result['structured_data'] = structured_data

        return result

    def _execute_multi_file_summary(self, question: str, plan: QueryPlan) -> Dict[str, Any]:
        """Summarize multiple files"""
        rag_logger.info(f"📚 Summarizing {len(plan.target_files)} files")

        summaries = []

        for filename in plan.target_files:
            # Get chunks for this file
            results = self.collection.get(where={"filename": filename})

            if not results['documents']:
                continue

            # Determine file type
            is_excel = any(m.get('content_type') == 'excel_metadata'
                          for m in results['metadatas'])

            if is_excel:
                # Excel summary - conversational
                metadata = results['metadatas'][0]
                summary = f"{filename} is an Excel file with {metadata.get('row_count', '?')} rows and {metadata.get('col_count', '?')} columns. It has columns like: {metadata.get('columns', 'N/A')}"
            else:
                # Text document - content summary
                content = '\n\n'.join(results['documents'][:5])  # First 5 chunks
                summary = f"{filename}: {self._quick_summarize(content)}"

            summaries.append(summary)

        # Combine all summaries conversationally
        if len(plan.target_files) == 1:
            intro = "Here's what I found in your document:"
        else:
            intro = f"Alright! You've got {len(plan.target_files)} files here. Let me give you a quick overview:"

        combined = '\n\n'.join(summaries)

        final_answer = f"""{intro}

{combined}

That's the gist of it! Feel free to ask me specific questions about any of these files."""

        return {
            'answer': final_answer,
            'sources': [{'filename': f} for f in plan.target_files],
            'has_context': True
        }

    def _execute_summary(self, question: str, plan: QueryPlan) -> Dict[str, Any]:
        """Summarize single file"""
        filename = plan.target_files[0]

        results = self.collection.get(where={"filename": filename})

        if not results['documents']:
            return {
                'answer': f"Hmm, I couldn't find any content for {filename}. Maybe it wasn't uploaded correctly?",
                'sources': [],
                'has_context': False
            }

        # Check file type
        is_excel = any(m.get('content_type') == 'excel_metadata'
                      for m in results['metadatas'])

        if is_excel:
            # Excel structure summary - conversational
            metadata = results['metadatas'][0]
            answer = f"""So, {filename} is an Excel spreadsheet with some nice structured data!

It's got {metadata.get('row_count', 'Unknown'):,} rows and {metadata.get('col_count', 'Unknown')} columns.

The columns are: {metadata.get('columns', 'N/A')}

You can ask me things like "What's the total of [column]?", "How many rows have [keyword]?", or "Show me entries from [date]" - I'll dig into the data for you!"""
        else:
            # Text document summary
            content = '\n\n'.join(results['documents'][:10])
            answer = self._generate_summary(content, filename)

        return {
            'answer': answer,
            'sources': [{'filename': filename}],
            'has_context': True
        }

    def _execute_retrieval(self, question: str, plan: QueryPlan) -> Dict[str, Any]:
        """Standard retrieval-based query with global search and citations"""
        # Search across target files
        query_embedding = self.generate_embedding(question, "retrieval_query")

        # Separate Excel and non-Excel files
        excel_target_files = []
        non_excel_target_files = []

        if plan.target_files:
            for filename in plan.target_files:
                if filename.lower().endswith(('.xlsx', '.xls', '.csv')):
                    excel_target_files.append(filename)
                else:
                    non_excel_target_files.append(filename)

        # === SEARCH NON-EXCEL FILES (Global Search) ===
        non_excel_contexts = []
        
        # Build where clause
        where_clause = {}
        if non_excel_target_files:
            if len(non_excel_target_files) == 1:
                where_clause = {"filename": non_excel_target_files[0]}
            else:
                where_clause = {"filename": {"$in": non_excel_target_files}}
        
        # Execute global query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=10,  # Get top 10 most relevant chunks across ALL files
            where=where_clause if where_clause else None
        )

        # Process results
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                non_excel_contexts.append({
                    'content': doc,
                    'filename': metadata['filename'],
                    'page': metadata.get('page_number', 'N/A')
                })

        # === SEARCH EXCEL FILES WITH CODE EXECUTION ===
        excel_results = {}
        
        if excel_target_files:
            # Extract keywords for searching
            keywords = self.query_router.extract_keywords(question)
            rag_logger.info(f"🔍 Searching Excel files for keywords: {keywords}")

            for filename in excel_target_files:
                # Get file path
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    where={
                        "$and": [
                            {"filename": filename},
                            {"content_type": "excel_metadata"}
                        ]
                    }
                )

                if results['metadatas'] and results['metadatas'][0]:
                    metadata = results['metadatas'][0][0]
                    file_path = metadata.get('file_path')

                    if file_path:
                        # Search Excel with code
                        excel_search_result = self._search_excel_for_keywords(file_path, filename, keywords)

                        if excel_search_result['found']:
                            excel_results[filename] = excel_search_result

        # === COMBINE RESULTS ===
        if not non_excel_contexts and not excel_results:
            return {
                'answer': "I couldn't find any relevant information in your documents to answer that question.",
                'sources': [],
                'has_context': False
            }

        # Generate answer with citations
        answer, sources_map = self._generate_hybrid_answer(question, non_excel_contexts, excel_results)

        result = {
            'answer': answer,
            'sources': sources_map,
            'has_context': True
        }

        # Add structured data from Excel if available
        if excel_results:
            structured_tables = []
            for filename, excel_data in excel_results.items():
                if excel_data['data']:
                    structured_tables.append({
                        'type': 'table',
                        'filename': filename,
                        'row_count': excel_data['count'],
                        'columns': list(excel_data['data'][0].keys()) if excel_data['data'] else [],
                        'data': excel_data['data']
                    })

            if structured_tables:
                result['structured_data'] = structured_tables

        return result

    # ===== HELPER METHODS =====
    
    def _search_excel_for_keywords(self, file_path: str, filename: str,
                                   keywords: List[str]) -> Dict[str, Any]:
        """Search Excel file for keywords using code execution"""
        # Build search code
        keyword_conditions = " | ".join([
            f"df.astype(str).apply(lambda row: row.str.contains('{kw}', case=False, na=False).any(), axis=1)"
            for kw in keywords
        ])

        code = f"""
import pandas as pd

# Load the data
file_path = r"{file_path}"
if file_path.endswith('.csv'):
    df = pd.read_csv(file_path)
else:
    df = pd.read_excel(file_path)

# Search for keywords across all columns
mask = {keyword_conditions}
filtered = df[mask]

# Return results
if len(filtered) > 0:
    result = {{
        'count': len(filtered),
        'data': filtered.head(10).to_dict('records')  # First 10 matches
    }}
else:
    result = {{'count': 0, 'data': []}}

print(f"Found {{len(filtered)}} matching rows")
"""

        rag_logger.info(f"🔍 Searching {filename} for: {keywords}")

        # Execute the search
        exec_result = self.excel_executor.execute_code(code, file_path)

        if exec_result['success'] and exec_result['result']:
            result_data = exec_result['result']
            return {
                'found': result_data.get('count', 0) > 0,
                'count': result_data.get('count', 0),
                'data': result_data.get('data', []),
                'filename': filename
            }
        else:
            return {'found': False, 'count': 0, 'data': [], 'filename': filename}

    def _generate_hybrid_answer(self, question: str,
                               pdf_contexts: List[Dict],
                               excel_results: Dict[str, Dict]) -> Tuple[str, List[Dict]]:
        """Generate answer with Perplexity-style citations"""

        # Build context with indexed sources
        context_parts = []
        sources_map = []
        source_index = 1

        # Add PDF contexts
        for ctx in pdf_contexts:
            context_parts.append(f"Source [{source_index}] (File: {ctx['filename']}):\n{ctx['content']}")
            sources_map.append({'id': source_index, 'filename': ctx['filename']})
            source_index += 1

        # Add Excel results
        for filename, result in excel_results.items():
            excel_content = f"Found {result['count']} matching rows in {filename}.\n"
            if result['data']:
                for i, row in enumerate(result['data'][:5], 1):
                    excel_content += f"Row {i}: " + ", ".join([f"{k}: {v}" for k, v in row.items()]) + "\n"
            
            context_parts.append(f"Source [{source_index}] (File: {filename}):\n{excel_content}")
            sources_map.append({'id': source_index, 'filename': filename})
            source_index += 1

        combined_context = "\n\n".join(context_parts)

        prompt = f"""Answer the user's question based ONLY on the provided sources.
        
**Question:** {question}

**Sources:**
{combined_context}

**Instructions:**
1. **Synthesize** the answer from multiple sources. Do NOT go file-by-file (e.g., don't say "In file X... then in file Y...").
2. **Cite your sources** using the format [1], [2], etc. immediately after the relevant information.
3. **Be concise and direct**.
4. If the sources conflict, mention the discrepancy.
5. If the answer is not in the sources, say so.
6. Use a professional but conversational tone.

**Example:**
The project revenue increased by 20% in Q1 [1], primarily driven by new product launches [2]. However, expenses also rose due to marketing costs [1].

Answer:"""

        try:
            rag_logger.info("=" * 70)
            rag_logger.info("📤 CITATION PROMPT:")
            rag_logger.info(prompt)
            rag_logger.info("=" * 70)

            response = self.model.generate_content(prompt)

            rag_logger.info("=" * 70)
            rag_logger.info("📥 CITATION ANSWER:")
            rag_logger.info(response.text)
            rag_logger.info("=" * 70)

            return response.text.strip(), sources_map

        except Exception as e:
            rag_logger.error(f"Hybrid answer generation error: {e}")
            return "I encountered an error generating the answer.", []

    def _get_available_files(self) -> List[Dict[str, Any]]:
        """Get list of available files with metadata"""
        docs = self.list_documents()

        available = []
        for doc in docs:
            file_info = {
                'filename': doc['filename'],
                'type': 'excel' if doc.get('content_type') == 'excel_metadata' else 'document'
            }

            if 'row_count' in doc:
                file_info['row_count'] = doc['row_count']
                file_info['col_count'] = doc['col_count']

            available.append(file_info)

        return available

    def _generate_analysis_code(self, question: str, filename: str,
                                file_metadata: Dict) -> Optional[str]:
        """Generate Python code for Excel analysis"""
        sample_preview = str(file_metadata['sample_data'])

        prompt = f"""You are writing Python code to analyze a pandas DataFrame.

**CRITICAL - DataFrame Operations ONLY:**
- Variable 'df' is a pandas DataFrame (NOT a list, NOT an array)
- Use DataFrame methods: df[condition], df['column'].method(), df.loc[], etc.
- DO NOT iterate with for loops like: for row in df
- DO NOT use list comprehensions on df
- DO NOT import anything
- DO NOT try to load the file

**Your Data:**
DataFrame 'df' has {file_metadata['row_count']:,} rows with these columns:
{', '.join(file_metadata['columns'])}

**Sample rows (showing actual data format):**
{sample_preview}

**Question:** {question}

**IMPORTANT - Searching Text:**
When searching for names/keywords in text:
```python
# FLEXIBLE search (recommended for names):
df['Remarks'].str.contains('vijay', case=False, na=False)  # Finds "vijay" anywhere
# OR combine multiple keywords:
mask = df['Remarks'].str.contains('vijay', case=False, na=False) | df['Remarks'].str.contains('patil', case=False, na=False)

# EXACT phrase search (less flexible):
df['Remarks'].str.contains('vijay patil', case=False, na=False)  # Needs exact "vijay patil"
```

**IMPORTANT - What to return:**
- If question asks "how many" / "count" → return: len(filtered_df)
- If question asks "show" / "find" / "list" / "transactions" / "details" / "all" → return: filtered_df.to_dict('records')
- If question asks for calculation → return: sum/mean/etc

**CRITICAL - Return ALL rows, not just a sample:**
- Do NOT use .head() or .head(10) or any limiting
- Return the COMPLETE filtered DataFrame as records
- We need ALL matching rows, not just first few

**Your Code Format:**
```python
# Step 1: Analyze the question type
print(f"Analyzing {{len(df)}} rows")

# Step 2: Search flexibly (use OR for multiple keywords)
mask = df['Remarks'].str.contains('keyword1', case=False, na=False) | df['Remarks'].str.contains('keyword2', case=False, na=False)
filtered = df[mask]

# Step 3: Return appropriate result
# If asking for details/transactions (return ALL rows, no .head()):
result = filtered.to_dict('records')  # ALL matching rows
# If asking for count:
result = len(filtered)
# If asking for sum:
result = filtered['Amount'].sum()

print(f"Found {{len(filtered)}} matching rows")
print(f"Answer: {{result}}")
```

Write ONLY Python code in ```python block (no explanations):"""

        try:
            rag_logger.info("=" * 70)
            rag_logger.info("📤 PROMPT TO AI:")
            rag_logger.info(prompt)
            rag_logger.info("=" * 70)

            response = self.model.generate_content(prompt)

            rag_logger.info("=" * 70)
            rag_logger.info("📥 AI RESPONSE:")
            rag_logger.info(response.text)
            rag_logger.info("=" * 70)

            return extract_python_code(response.text)
        except Exception as e:
            rag_logger.error(f"Code generation error: {e}")
            return None

    def _fix_code(self, broken_code: str, error: str, metadata: Dict) -> Optional[str]:
        """Fix broken code"""
        prompt = f"""Fix this Python code that failed with an error.

**Error:** {error}

**Your broken code:**
```python
{broken_code}
```

**CRITICAL - DataFrame Operations:**
- 'df' is a pandas DataFrame (already loaded)
- Use: df['ColumnName'], df[df['Column'] == value], df.loc[], etc.
- DO NOT: for row in df, [x for x in df], pd.read_excel()
- Available columns: {metadata['columns']}

**Common fixes:**
1. FileNotFoundError → Don't load file, use 'df' directly
2. KeyError → Check column names are EXACT (case-sensitive): {metadata['columns']}
3. TypeError (iteration) → Don't iterate df, use DataFrame operations
4. NameError → Define variables before using them

**Example correct DataFrame operations:**
```python
# Filter
filtered = df[df['Date'] == '20/03/2025']
# Count
result = len(filtered)
# Sum
result = df['Amount'].sum()
```

Output ONLY fixed Python code in ```python block:"""

        try:
            rag_logger.info("=" * 70)
            rag_logger.info("🔧 FIX PROMPT TO AI:")
            rag_logger.info(prompt)
            rag_logger.info("=" * 70)

            response = self.model.generate_content(prompt)

            rag_logger.info("=" * 70)
            rag_logger.info("📥 AI FIX RESPONSE:")
            rag_logger.info(response.text)
            rag_logger.info("=" * 70)

            return extract_python_code(response.text)
        except:
            return None

    def _format_excel_answer(self, exec_result: Dict, code: str,
                            filename: str, metadata: Dict) -> Tuple[str, Optional[Dict]]:
        """Format successful Excel analysis answer in conversational style

        Returns:
            Tuple of (conversational_answer, structured_data_dict or None)
        """
        result = exec_result['result']
        structured_data = None

        # Prepare structured data summary for AI
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            # We have actual transaction data
            data_summary = f"Found {len(result)} matching transactions. "

            # Format sample data for AI (first 5 only)
            sample_transactions = ""
            for i, row in enumerate(result[:5], 1):
                sample_transactions += f"\nTransaction {i}:\n"
                for key, value in row.items():
                    sample_transactions += f"  {key}: {value}\n"

            if len(result) > 5:
                data_summary += f"(Showing 5 sample transactions out of {len(result)} total)"

            # Prepare structured data for frontend (ALL rows)
            # Frontend expects an ARRAY of table objects
            structured_data = [{
                'filename': filename,
                'columns': list(result[0].keys()) if result else [],
                'data': result  # ALL data, not limited
            }]

            result_type = "transaction_list"
        else:
            # We have a single value (count, sum, etc.)
            data_summary = f"Result: {result}"
            sample_transactions = ""
            result_type = "single_value"

        # Use AI to create conversational response
        prompt = f"""Convert this data analysis result into a friendly, conversational response as if you're explaining it to a friend.

**User's Question Context:** The user asked a question about their {filename} file.

**Analysis Result:**
{data_summary}

**Detailed Data:**
{sample_transactions if sample_transactions else f"The answer is: {result}"}

**Instructions:**
- Write in a natural, friendly, conversational tone (like talking to a friend)
- Don't use markdown headers (no ##, ###) or bold text (**text**)
- Start with a friendly opener like "Hey!", "Alright!", "So...", etc.
- Explain the result naturally
- If there are multiple transactions, mention the count and give a brief summary
- Keep it SHORT and friendly - don't list all transactions (we'll show them in a table)
- Don't mention technical terms like "DataFrame", "code execution", etc.
- Just present the summary in a natural way

Example good response:
"Hey! I found 15 transactions related to Vijay Patil in your bank statement. They span from March to April, with amounts ranging from ₹50 to ₹1,500. You can see all the details in the table below!"

Now write your conversational response:"""

        try:
            rag_logger.info("=" * 70)
            rag_logger.info("📤 CONVERSATIONAL FORMATTING PROMPT:")
            rag_logger.info(prompt)
            rag_logger.info("=" * 70)

            response = self.model.generate_content(prompt)

            rag_logger.info("=" * 70)
            rag_logger.info("📥 CONVERSATIONAL RESPONSE:")
            rag_logger.info(response.text)
            rag_logger.info("=" * 70)

            conversational_answer = response.text.strip()

            # Add source info naturally
            conversational_answer += f"\n\n(Data from {filename} - {metadata['row_count']:,} total rows)"

            return conversational_answer, structured_data

        except Exception as e:
            rag_logger.error(f"Conversational formatting error: {e}")
            # Fallback to basic format
            return f"Found the answer! The result is: {result}\n\n(From {filename})", structured_data

    def _format_error_answer(self, exec_result: Dict, filename: str) -> str:
        """Format error answer in conversational style"""
        return f"""Hmm, I ran into a problem while analyzing {filename}.

The issue was: {exec_result['error']}

Could you try rephrasing your question? Or maybe double-check if you're asking about the right columns? I'm here to help!"""

    def _quick_summarize(self, content: str) -> str:
        """Quick summary of content"""
        lines = content.split('\n')[:10]
        preview = '\n'.join(lines)
        return f"Preview: {preview[:500]}..."

    def _generate_summary(self, content: str, filename: str) -> str:
        """Generate AI summary of content in conversational style"""
        prompt = f"""Summarize this document in a friendly, conversational way as if you're telling a friend about it.

**File:** {filename}

**Content:**
{content[:3000]}

**Instructions:**
- Write in a natural, friendly tone
- Don't use markdown headers (no ##)
- Start naturally like "So this document is about..." or "Alright, {filename} contains..."
- Be concise but informative (2-4 sentences)
- Make it feel like a conversation, not a formal summary

Your conversational summary:"""

        try:
            rag_logger.info("=" * 70)
            rag_logger.info("📤 SUMMARY PROMPT:")
            rag_logger.info(prompt)
            rag_logger.info("=" * 70)

            response = self.model.generate_content(prompt)

            rag_logger.info("=" * 70)
            rag_logger.info("📥 SUMMARY RESPONSE:")
            rag_logger.info(response.text)
            rag_logger.info("=" * 70)

            return response.text.strip()
        except Exception as e:
            rag_logger.error(f"Summary generation error: {e}")
            return f"I've got {filename} here. Feel free to ask me any questions about it!"

    def _generate_answer_from_context(self, question: str,
                                     context_texts: List[str],
                                     sources: List[str]) -> str:
        """Generate answer from retrieved context in conversational style"""

        # Check if we have multiple sources
        has_multiple_sources = len(sources) > 1

        if has_multiple_sources:
            # Build context with clear source attribution
            context_with_sources = ""
            for ctx in context_texts:
                context_with_sources += f"\n{ctx}\n---"

            prompt = f"""Answer the question based on the provided context from MULTIPLE files. Write in a friendly, conversational tone.

**Context from multiple documents:**
{context_with_sources}

**Question:** {question}

**IMPORTANT - Multiple Sources:**
You have information from {len(sources)} different files: {', '.join(sources)}
- When presenting information, CLEARLY INDICATE which file it came from
- Structure your answer to show information from each source
- Example format:
  "Hey! I found information about this in a couple of places:

  From your Excel file (bank_statement.xlsx), I can see...

  And from your PDF (report.pdf), it mentions..."

**Instructions:**
- Answer ONLY from the context provided
- Write in a natural, friendly, conversational tone (like talking to a friend)
- Don't use markdown headers (no ##, ###) or excessive bold text
- Start naturally, maybe with "So...", "Alright...", "Hey...", etc.
- CLEARLY show which information came from which file
- Be specific and include relevant details from the context
- Keep it informative but friendly

Answer:"""
        else:
            # Single source - simpler prompt
            context = '\n\n'.join(context_texts)

            prompt = f"""Answer the question based on the provided context. Write in a friendly, conversational tone as if you're explaining it to a friend.

**Context from {sources[0]}:**
{context}

**Question:** {question}

**Instructions:**
- Answer ONLY from the context provided
- Write in a natural, friendly, conversational tone (like talking to a friend)
- Don't use markdown headers (no ##, ###) or excessive bold text
- Start naturally, maybe with "So...", "Alright...", "Hey...", etc.
- Be specific and include relevant details from the context
- If the answer is not in the context, say so naturally (like "Hmm, I couldn't find that information")
- Don't mention "the context says" - just explain naturally
- Keep it informative but friendly

Answer:"""

        try:
            rag_logger.info("=" * 70)
            rag_logger.info("📤 CONVERSATIONAL ANSWER PROMPT:")
            rag_logger.info(prompt)
            rag_logger.info("=" * 70)

            response = self.model.generate_content(prompt)

            rag_logger.info("=" * 70)
            rag_logger.info("📥 CONVERSATIONAL ANSWER:")
            rag_logger.info(response.text)
            rag_logger.info("=" * 70)

            answer = response.text.strip()

            # Only add sources at the end if AI didn't already mention them
            if has_multiple_sources and not any(src in answer for src in sources):
                answer += f"\n\n(Found this across: {', '.join(sources)})"
            elif not has_multiple_sources and sources[0] not in answer:
                answer += f"\n\n(From {sources[0]})"

            return answer

        except Exception as e:
            rag_logger.error(f"Answer generation error: {e}")
            return "Hmm, I ran into an issue generating the answer. Could you try asking again?"

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        docs = self.list_documents()

        total_chunks = sum(doc.get('chunks', 0) for doc in docs)

        # Calculate total words (estimate from chunks)
        total_words = total_chunks * 400  # Rough estimate: 400 words per chunk

        return {
            'num_documents': len(docs),
            'total_documents': len(docs),  # For frontend compatibility
            'total_chunks': total_chunks,
            'total_words': total_words,
            'collection_name': self.collection.name,
            'documents': docs
        }

    def reset_collection(self):
        """Reset entire collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(self.collection.name)
        rag_logger.info("🗑️ Collection reset")
