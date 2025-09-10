#!/usr/bin/env python3
"""
Flask Web Interface for LLM-driven Intelligent RAG System
LLM智能RAG系统的Flask Web界面
"""

import os
import time
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from enhanced_rag_system import EnhancedRAGSystem

# 加载环境变量
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# 全局RAG系统实例
rag_system = None

def initialize_rag_system():
    """初始化RAG系统"""
    global rag_system
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY 环境变量未设置"
    
    try:
        rag_system = EnhancedRAGSystem(
            openai_api_key=api_key,
            chroma_path="./chroma",
            model="gpt-4o-mini"
        )
        return True, "系统初始化成功"
    except Exception as e:
        return False, f"系统初始化失败: {str(e)}"

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """执行搜索"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'success': False, 'error': '查询不能为空'})
        
        if rag_system is None:
            return jsonify({'success': False, 'error': '系统未初始化'})
        
        # 执行搜索
        initial_k = data.get('initial_k', 8)
        final_k = data.get('final_k', 3)
        use_llm_ranking = data.get('use_llm_ranking', True)
        
        start_time = time.time()
        result = rag_system.query(
            query_text=query,
            initial_k=initial_k,
            final_k=final_k,
            use_llm_ranking=use_llm_ranking
        )
        
        # 格式化结果
        response = {
            'success': True,
            'query': query,
            'answer': result['answer'],
            'processing_time': result['processing_time'],
            'vector_search_time': result['vector_search_time'],
            'llm_ranking_time': result['llm_ranking_time'],
            'answer_generation_time': result['answer_generation_time'],
            'highlighted_evidence': result.get('highlighted_evidence', ''),
            'chunks': []
        }
        
        # 处理chunks
        for i, chunk in enumerate(result.get('chunks', []), 1):
            chunk_info = {
                'index': i,
                'final_score': chunk.get('final_score', 0),
                'similarity_score': chunk.get('similarity_score', 0),
                'llm_score': chunk.get('llm_score', 0),
                'content': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                'full_content': chunk['content'],
                'highlighted_content': chunk.get('highlighted_content', chunk['content']),
                'relevance_reason': chunk.get('relevance_reason', '无分析')
            }
            response['chunks'].append(chunk_info)
        
        # 排序摘要
        ranking_summary = result.get('ranking_summary', {})
        if isinstance(ranking_summary, dict):
            response['ranking_summary'] = ranking_summary.get('ranking_summary', '')
            response['llm_improvement'] = ranking_summary.get('llm_improvement', 0)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/system_info')
def system_info():
    """获取系统信息"""
    try:
        if rag_system is None:
            return jsonify({'initialized': False})
        
        return jsonify({
            'initialized': True,
            'status': '运行正常',
            'model': 'gpt-4o-mini',
            'dataset': 'single_20240229.json (9,103条)',
            'embedding_model': 'text-embedding-ada-002'
        })
        
    except Exception as e:
        return jsonify({'initialized': False, 'error': str(e)})

# 创建模板目录和文件
def create_templates():
    """创建HTML模板"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 LLM智能RAG系统</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .search-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }
        input, textarea, select {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .results {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            font-size: 12px;
        }
        .answer {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
            margin: 20px 0;
        }
        .chunk {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .chunk-header {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .chunk-scores {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
        }
        .score {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .score-final { background: #667eea; color: white; }
        .score-vector { background: #28a745; color: white; }
        .score-llm { background: #fd7e14; color: white; }
        .highlighted {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .examples {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .example-btn {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 8px 12px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.2s;
        }
        .example-btn:hover {
            background: #e9ecef;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 LLM智能RAG系统</h1>
        <p>搭载GPT-4o-mini智能ranking和highlighting功能</p>
    </div>

    <div class="search-section">
        <div class="input-group">
            <label for="query">📝 请输入您的问题:</label>
            <textarea id="query" rows="3" placeholder="例: コンバインについて教えて"></textarea>
        </div>

        <div class="examples">
            <div class="example-btn" onclick="setExample('コンバインとは何ですか？')">コンバインとは？</div>
            <div class="example-btn" onclick="setExample('音位転倒について説明してください')">音位転倒について</div>
            <div class="example-btn" onclick="setExample('農業機械の種類について教えて')">農業機械について</div>
            <div class="example-btn" onclick="setExample('慣用句の間違いを指摘する方法')">慣用句の間違い</div>
            <div class="example-btn" onclick="setExample('待ち合わせのマナーについて')">待ち合わせマナー</div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 20px 0;">
            <div class="input-group">
                <label for="initial_k">🔍 初始检索数量:</label>
                <select id="initial_k">
                    <option value="5">5</option>
                    <option value="8" selected>8</option>
                    <option value="10">10</option>
                </select>
            </div>
            <div class="input-group">
                <label for="final_k">📊 最终返回数量:</label>
                <select id="final_k">
                    <option value="2">2</option>
                    <option value="3" selected>3</option>
                    <option value="5">5</option>
                </select>
            </div>
            <div class="input-group">
                <label for="use_llm_ranking">🧠 启用LLM智能排序:</label>
                <select id="use_llm_ranking">
                    <option value="true" selected>是</option>
                    <option value="false">否</option>
                </select>
            </div>
        </div>

        <button onclick="performSearch()" id="searchBtn">🔍 开始智能检索</button>
    </div>

    <div id="results"></div>

    <script>
        function setExample(text) {
            document.getElementById('query').value = text;
        }

        async function performSearch() {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('请输入查询内容');
                return;
            }

            const searchBtn = document.getElementById('searchBtn');
            const resultsDiv = document.getElementById('results');

            searchBtn.disabled = true;
            searchBtn.textContent = '🧠 LLM智能分析中...';

            resultsDiv.innerHTML = '<div class="loading">🧠 正在使用LLM进行智能分析，请稍候...</div>';

            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        initial_k: parseInt(document.getElementById('initial_k').value),
                        final_k: parseInt(document.getElementById('final_k').value),
                        use_llm_ranking: document.getElementById('use_llm_ranking').value === 'true'
                    })
                });

                const result = await response.json();

                if (result.success) {
                    displayResults(result);
                } else {
                    resultsDiv.innerHTML = `<div class="error">❌ 错误: ${result.error}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">❌ 网络错误: ${error.message}</div>`;
            } finally {
                searchBtn.disabled = false;
                searchBtn.textContent = '🔍 开始智能检索';
            }
        }

        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            
            let html = '<div class="results">';
            
            // 性能指标
            html += '<div class="metrics">';
            html += `<div class="metric"><div class="metric-value">${result.processing_time.toFixed(2)}s</div><div class="metric-label">⏱️ 总处理时间</div></div>`;
            html += `<div class="metric"><div class="metric-value">${result.vector_search_time.toFixed(2)}s</div><div class="metric-label">📊 向量检索</div></div>`;
            html += `<div class="metric"><div class="metric-value">${result.llm_ranking_time.toFixed(2)}s</div><div class="metric-label">🧠 LLM排序</div></div>`;
            html += `<div class="metric"><div class="metric-value">${result.answer_generation_time.toFixed(2)}s</div><div class="metric-label">💬 答案生成</div></div>`;
            html += '</div>';

            // AI回答
            html += '<h2>💬 AI智能回答</h2>';
            html += `<div class="answer"><strong>问题:</strong> ${result.query}<br><br><strong>回答:</strong><br>${result.answer}</div>`;

            // 高亮证据
            if (result.highlighted_evidence) {
                html += '<h2>🔦 LLM高亮显示证据</h2>';
                html += `<div class="highlighted">${result.highlighted_evidence}</div>`;
            }

            // 详细结果
            if (result.chunks && result.chunks.length > 0) {
                html += `<h2>📊 详细检索结果 (${result.chunks.length}个chunks)</h2>`;
                
                result.chunks.forEach(chunk => {
                    html += '<div class="chunk">';
                    html += `<div class="chunk-header">📄 Chunk ${chunk.index}</div>`;
                    html += '<div class="chunk-scores">';
                    html += `<span class="score score-final">最终: ${chunk.final_score.toFixed(3)}</span>`;
                    html += `<span class="score score-vector">向量: ${chunk.similarity_score.toFixed(3)}</span>`;
                    html += `<span class="score score-llm">LLM: ${chunk.llm_score.toFixed(3)}</span>`;
                    html += '</div>';
                    
                    html += '<h4>📝 原始内容:</h4>';
                    html += `<p>${chunk.content}</p>`;
                    
                    html += '<h4>🔦 LLM高亮内容:</h4>';
                    html += `<div class="highlighted">${chunk.highlighted_content}</div>`;
                    
                    html += '<h4>💭 LLM相关性分析:</h4>';
                    html += `<p style="color: #666; font-size: 14px;">${chunk.relevance_reason}</p>`;
                    
                    html += '</div>';
                });
            }

            // 排序分析
            if (result.ranking_summary) {
                html += '<h2>📈 LLM排序分析</h2>';
                html += `<pre style="background: #f8f9fa; padding: 15px; border-radius: 8px; white-space: pre-wrap;">${result.ranking_summary}</pre>`;
                
                if (result.llm_improvement !== undefined) {
                    const improvement = result.llm_improvement;
                    if (improvement > 0) {
                        html += `<div style="color: #28a745; font-weight: bold;">🎯 LLM改进: +${improvement.toFixed(3)} (LLM排序效果更好)</div>`;
                    } else if (improvement < 0) {
                        html += `<div style="color: #ffc107; font-weight: bold;">📉 LLM改进: ${improvement.toFixed(3)} (向量排序更优)</div>`;
                    } else {
                        html += `<div style="color: #6c757d; font-weight: bold;">🟰 LLM改进: ${improvement.toFixed(3)} (效果相当)</div>`;
                    }
                }
            }

            html += '</div>';
            resultsDiv.innerHTML = html;
        }

        // 回车键搜索
        document.getElementById('query').addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                performSearch();
            }
        });
    </script>
</body>
</html>'''

    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    print("🚀 启动LLM智能RAG系统Web界面...")
    
    # 创建模板
    create_templates()
    
    # 初始化RAG系统
    success, message = initialize_rag_system()
    if success:
        print(f"✅ {message}")
        print("🌐 启动Web服务器...")
        print("📱 请在浏览器中访问: http://localhost:8501")
        print("⚡ 系统特性:")
        print("  - 🧠 LLM智能ranking")
        print("  - 🔦 智能高亮显示")
        print("  - 📊 详细性能分析")
        print("  - 💬 GPT-4o-mini驱动")
        
        app.run(host='0.0.0.0', port=8501, debug=False)
    else:
        print(f"❌ {message}")