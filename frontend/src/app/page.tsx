"use client";

import { useState, useEffect } from 'react';
import { Upload, FileText, Trash2, Loader2, Send, Bot, User, Database, Layers, Info } from 'lucide-react';
import { API_ENDPOINTS, API_BASE_URL } from '@/lib/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{ id: number; filename: string }>;
  structured_data?: any[];
}

interface Document {
  filename: string;
  chunks: number;
  file_type: string;
}

export default function Home() {
  // Upload state
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  // Documents state
  const [documents, setDocuments] = useState<Document[]>([]);
  const [stats, setStats] = useState({ total_documents: 0, total_chunks: 0, vector_db_size: 0 });
  const [deletingFile, setDeletingFile] = useState<string | null>(null);

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Fetch documents and stats
  const fetchData = async () => {
    try {
      const [docsRes, statsRes] = await Promise.all([
        fetch(API_ENDPOINTS.documents),
        fetch(API_ENDPOINTS.stats)
      ]);
      const docsData = await docsRes.json();
      const statsData = await statsRes.json();
      setDocuments(docsData.documents || []);
      setStats(statsData);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    }
  };

  // Upload handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFile(files[0]);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadFile(e.target.files[0]);
    }
  };

  const uploadFile = async (file: File) => {
    setIsUploading(true);
    setUploadStatus(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(API_ENDPOINTS.upload, { method: 'POST', body: formData });
      const result = await response.json();
      if (response.ok) {
        setUploadStatus({ type: 'success', message: `${result.filename} uploaded (${result.chunks_created} chunks)` });
        fetchData();
      } else {
        throw new Error(result.detail || 'Upload failed');
      }
    } catch (error: any) {
      setUploadStatus({ type: 'error', message: error.message });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (filename: string) => {
    if (!confirm(`Delete ${filename}?`)) return;
    setDeletingFile(filename);
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      if (response.ok) {
        fetchData();
      }
    } catch (error) {
      console.error('Delete error:', error);
    } finally {
      setDeletingFile(null);
    }
  };

  // Chat handlers
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(API_ENDPOINTS.query, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage }),
      });
      const data = await response.json();
      if (response.ok) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.answer, sources: data.sources, structured_data: data.structured_data }]);
      } else {
        throw new Error(data.detail || 'Failed to get response');
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I encountered an error. Please try again." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderContent = (content: string, sources: Array<{ id: number; filename: string }> = []) => {
    const parts = content.split(/(\[\d+\])/g);
    return parts.map((part, index) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const id = parseInt(match[1]);
        const source = sources.find(s => s.id === id);
        if (source) {
          return (
            <span key={index} className="inline-flex items-center justify-center w-5 h-5 bg-gray-200 text-gray-700 rounded text-xs font-medium mx-0.5" title={source.filename}>
              {id}
            </span>
          );
        }
      }
      return <span key={index}>{part}</span>;
    });
  };

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">RAG Multi-File AI</h1>
          <div className="flex items-center gap-4 text-sm text-gray-600">
            <div className="flex items-center gap-1">
              <FileText className="w-4 h-4" />
              <span>{stats.total_documents || 0} docs</span>
            </div>
            <div className="flex items-center gap-1">
              <Layers className="w-4 h-4" />
              <span>{stats.total_chunks || 0} chunks</span>
            </div>
            <div className="flex items-center gap-1">
              <Database className="w-4 h-4" />
              <span>{(stats.vector_db_size || 0).toFixed(1)} MB</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-4">
        {/* Info Banner */}
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start gap-2 text-sm text-blue-900">
            <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <div>
              <strong>How it works:</strong> Upload documents (PDF, Excel, CSV, DOCX) → AI chunks & embeds them → Ask questions in natural language → Get answers with source citations.
              For Excel files, AI generates Python code to analyze your data.
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-4">
          {/* Left: Upload & Documents */}
          <div className="lg:col-span-1 space-y-4">
            {/* Upload */}
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h2 className="font-semibold text-gray-900 mb-3">Upload Documents</h2>
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition ${
                  isDragging ? 'border-gray-400 bg-gray-50' : 'border-gray-300 hover:border-gray-400'
                } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input')?.click()}
              >
                {isUploading ? (
                  <Loader2 className="w-8 h-8 text-gray-400 mx-auto animate-spin" />
                ) : (
                  <Upload className="w-8 h-8 text-gray-400 mx-auto" />
                )}
                <p className="mt-2 text-sm text-gray-600">
                  Drop files or <span className="text-gray-900 font-medium">browse</span>
                </p>
                <p className="text-xs text-gray-500 mt-1">PDF, DOCX, XLSX, CSV, JSON</p>
              </div>
              <input
                id="file-input"
                type="file"
                className="hidden"
                onChange={handleFileSelect}
                accept=".pdf,.docx,.txt,.xlsx,.json,.csv"
              />
              {uploadStatus && (
                <div className={`mt-3 p-2 rounded text-sm ${uploadStatus.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'}`}>
                  {uploadStatus.message}
                </div>
              )}
            </div>

            {/* Documents List */}
            <div className="bg-white rounded-lg border border-gray-200">
              <div className="p-4 border-b border-gray-200">
                <h2 className="font-semibold text-gray-900">Documents ({documents.length})</h2>
              </div>
              <div className="max-h-96 overflow-y-auto">
                {documents.length === 0 ? (
                  <div className="p-8 text-center text-gray-500 text-sm">No documents yet</div>
                ) : (
                  <div className="divide-y divide-gray-100">
                    {documents.map((doc) => (
                      <div key={doc.filename} className="p-3 hover:bg-gray-50 flex items-center justify-between group">
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <FileText className="w-4 h-4 text-gray-400 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 truncate">{doc.filename}</p>
                            <p className="text-xs text-gray-500">{doc.chunks} chunks • {(doc.file_type || 'unknown').toUpperCase()}</p>
                          </div>
                        </div>
                        <button
                          onClick={() => handleDelete(doc.filename)}
                          disabled={deletingFile === doc.filename}
                          className="p-1 text-red-500 hover:bg-red-50 rounded opacity-0 group-hover:opacity-100 transition"
                        >
                          {deletingFile === doc.filename ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right: Chat */}
          <div className="lg:col-span-2 bg-white rounded-lg border border-gray-200 flex flex-col" style={{ height: 'calc(100vh - 200px)' }}>
            <div className="p-4 border-b border-gray-200">
              <h2 className="font-semibold text-gray-900">Chat</h2>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
                  <Bot className="w-12 h-12 mb-3 text-gray-300" />
                  <p className="text-sm">Upload documents and ask questions</p>
                </div>
              )}

              {messages.map((msg, idx) => (
                <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'assistant' ? 'bg-gray-200' : 'bg-gray-800'}`}>
                    {msg.role === 'assistant' ? <Bot className="w-4 h-4 text-gray-600" /> : <User className="w-4 h-4 text-white" />}
                  </div>
                  <div className="flex-1 space-y-2">
                    <div className={`inline-block px-4 py-2 rounded-lg text-sm ${msg.role === 'assistant' ? 'bg-gray-100 text-gray-900' : 'bg-gray-800 text-white'}`}>
                      {msg.role === 'assistant' ? renderContent(msg.content, msg.sources) : msg.content}
                    </div>

                    {msg.sources && msg.sources.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {msg.sources.map((source, sourceIdx) => (
                          <div key={`${idx}-${source.id}-${sourceIdx}`} className="flex items-center gap-1 px-2 py-1 bg-gray-100 rounded text-xs text-gray-700">
                            <span className="w-4 h-4 bg-gray-200 rounded-full flex items-center justify-center text-[10px] font-medium">{source.id}</span>
                            <span>{source.filename}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {msg.structured_data && Array.isArray(msg.structured_data) && msg.structured_data.map((table: any, i: number) => (
                      <div key={i} className="border border-gray-200 rounded-lg overflow-hidden">
                        <div className="bg-gray-50 px-3 py-2 text-xs font-medium text-gray-700 border-b border-gray-200">
                          {table.filename || 'Data'}
                        </div>
                        <div className="overflow-x-auto max-h-64">
                          <table className="w-full text-xs">
                            <thead className="bg-gray-50">
                              <tr>
                                {Array.isArray(table.columns) && table.columns.map((col: string, k: number) => (
                                  <th key={k} className="px-3 py-2 text-left font-medium text-gray-700 border-b border-gray-200">{col}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {Array.isArray(table.data) && table.data.map((row: any, j: number) => (
                                <tr key={j} className="hover:bg-gray-50">
                                  {Array.isArray(table.columns) && table.columns.map((col: string, k: number) => (
                                    <td key={`${j}-${k}`} className="px-3 py-2 text-gray-600 border-b border-gray-100">{row[col]}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                    <Bot className="w-4 h-4 text-gray-600" />
                  </div>
                  <div className="flex gap-1 items-center px-4 py-2 bg-gray-100 rounded-lg">
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                  </div>
                </div>
              )}
            </div>

            <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask anything about your documents..."
                  disabled={isLoading}
                  className="flex-1 px-4 py-2 bg-gray-50 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-400 text-sm disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  className="px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition"
                >
                  {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
