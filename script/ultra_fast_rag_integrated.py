#!/usr/bin/env python3
"""
超高速RAG系统 - 整合版本
集成了LLM评分、精细chunking、向量搜索的完整系统
"""

import os
import re
import json
import time
from typing import Optional, Tuple, List, Dict, Any
from pydantic import SecretStr
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv


class IntegratedLLMEvidenceRanker:
    """内置LLM证据排序系统"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini", highlight_mode: str = "auto"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.highlight_mode = highlight_mode  # "auto", "sentence", "keyword"
    
    def _extract_precise_sentences(self, content: str, query: str) -> str:
        """智能高亮提取 - 支持关键词级别和句子级别"""
        import re
        
        # 提取查询关键词
        query_clean = re.sub(r'[とは何ですかについて]', '', query)
        keywords = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+', query_clean)
        keywords = [kw for kw in keywords if len(kw) >= 2]
        
        # 句子分割
        sentences = re.split(r'[。！？.!?]', content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        
        if not sentences:
            return content[:50] + "..." if len(content) > 50 else content
        
        # 计算每个句子的匹配分数
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # 关键词匹配
            for keyword in keywords:
                if keyword in sentence:
                    score += 2
            
            # 定义句子优先（包含"とは"、"は、"等）
            if any(marker in sentence for marker in ['とは', 'は、', 'です', 'である']):
                score += 3
            
            # 位置权重（前面的句子权重更高）
            score += (len(sentences) - i) * 0.1
            
            sentence_scores.append((sentence, score))
        
        # 选择最高分的句子
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        best_sentence = sentence_scores[0][0]
        
        # 根据模式选择高亮策略
        if self.highlight_mode == "sentence":
            # 强制句子模式
            if len(best_sentence) <= 120:
                return best_sentence + '。'
            else:
                return best_sentence[:100] + '...'
        elif self.highlight_mode == "keyword":
            # 强制关键词模式
            keyword_highlight = self._extract_keyword_highlight(best_sentence, keywords)
            if keyword_highlight:
                return keyword_highlight
            else:
                return best_sentence[:80] + '...'
        else:
            # 自动模式：智能选择
            if len(best_sentence) <= 60:
                # 短句子：直接返回
                return best_sentence + '。'
            elif len(best_sentence) <= 120:
                # 中等长度：尝试关键词高亮
                keyword_highlight = self._extract_keyword_highlight(best_sentence, keywords)
                if keyword_highlight and len(keyword_highlight) < len(best_sentence):
                    return keyword_highlight
                else:
                    return best_sentence + '。'
            else:
                # 长句子：强制关键词高亮
                keyword_highlight = self._extract_keyword_highlight(best_sentence, keywords)
                if keyword_highlight:
                    return keyword_highlight
                else:
                    # 回退到句子截断
                    return best_sentence[:100] + '...'
    
    def _extract_keyword_highlight(self, sentence: str, keywords: List[str]) -> str:
        """提取关键词周围的高亮文本"""
        if not keywords:
            return None
        
        # 找到所有关键词在句子中的位置
        keyword_positions = []
        for keyword in keywords:
            start = 0
            while True:
                pos = sentence.find(keyword, start)
                if pos == -1:
                    break
                keyword_positions.append((pos, pos + len(keyword), keyword))
                start = pos + 1
        
        if not keyword_positions:
            return None
        
        # 按位置排序
        keyword_positions.sort(key=lambda x: x[0])
        
        # 找到关键词的覆盖范围
        min_pos = min(pos[0] for pos in keyword_positions)
        max_pos = max(pos[1] for pos in keyword_positions)
        
        # 扩展上下文（前后各20个字符）
        context_start = max(0, min_pos - 20)
        context_end = min(len(sentence), max_pos + 20)
        
        highlighted_text = sentence[context_start:context_end]
        
        # 确保不截断单词
        if context_start > 0 and highlighted_text[0] not in '。！？.!? ':
            # 向前找到完整单词的开始
            while context_start > 0 and sentence[context_start] not in '。！？.!? ':
                context_start -= 1
            highlighted_text = sentence[context_start:context_end]
        
        if context_end < len(sentence) and highlighted_text[-1] not in '。！？.!? ':
            # 向后找到完整单词的结束
            while context_end < len(sentence) and sentence[context_end] not in '。！？.!? ':
                context_end += 1
            highlighted_text = sentence[context_start:context_end]
        
        # 添加省略号如果截断了
        if context_start > 0:
            highlighted_text = '...' + highlighted_text
        if context_end < len(sentence):
            highlighted_text = highlighted_text + '...'
        
        return highlighted_text
    
    def extract_single_sentence_forced(self, text: str) -> str:
        """强制提取单句 - 绝不失败的方法"""
        import re
        
        # 方法1：按句号分割
        sentences = re.split(r'[。！？.!?]', text)
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence and 10 <= len(clean_sentence) <= 100:
                return clean_sentence + "。"
        
        # 方法2：按逗号分割（如果句子太长）
        if sentences and sentences[0]:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 100:
                parts = first_sentence.split('、')
                if len(parts) > 1 and len(parts[0]) >= 20:
                    return parts[0] + "。"
        
        # 方法3：强制字符截断
        clean_text = text.strip()
        if len(clean_text) <= 100:
            return clean_text
        
        # 找到100字符内的最后一个合理分割点
        for i in range(min(80, len(clean_text)), min(100, len(clean_text))):
            if clean_text[i] in '、，。！？':
                return clean_text[:i+1]
        
        # 最后手段：强制截断
        return clean_text[:80] + "..."

    def rank_and_highlight_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 1) -> List[Dict[str, Any]]:
        """优化的排序方法 - 更快速度"""
        if not chunks:
            return []
        
        print(f"🎯 执行精确句子级LLM评估...")
        
        # 只对top 1个向量匹配最好的chunk进行LLM评估，提高速度
        chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        top_chunk = chunks[0]
        
        try:
            llm_score, relevance_reason, highlighted_content = self._evaluate_chunk_with_llm(
                query, top_chunk["content"], top_chunk.get("similarity_score", 0.0)
            )
            
            enhanced_chunk = {
                **top_chunk,
                "llm_score": llm_score,
                "relevance_reason": relevance_reason,
                "highlighted_content": highlighted_content,
                "final_score": (top_chunk.get("similarity_score", 0.0) * 0.4) + (llm_score * 0.6),
                "rank_order": 1
            }
            
            print(f"✅ 精确提取完成: {len(highlighted_content)}字符 (vs 原始{len(top_chunk['content'])}字符)")
            
            return [enhanced_chunk]
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            # 确保回退机制
            highlighted_content = self._extract_precise_sentences(top_chunk["content"], query)
            enhanced_chunk = {
                **top_chunk,
                "llm_score": top_chunk.get("similarity_score", 0.0),
                "relevance_reason": "回退到本地精确提取",
                "highlighted_content": highlighted_content,
                "final_score": top_chunk.get("similarity_score", 0.0),
                "rank_order": 1
            }
            return [enhanced_chunk]
    
    def _evaluate_chunk_with_llm(self, query: str, content: str, vector_score: float) -> Tuple[float, str, str]:
        """改进的LLM评估 - GPT生成完整回答"""

        # 让GPT根据文本内容生成完整回答
        answer_generation_prompt = f"""
查询: {query}
参考文本: {content}

请根据参考文本回答用户的查询。要求:
1. 直接回答用户的问题
2. 基于参考文本内容，但用你自己的话来表达
3. 回答要完整、准确、简洁
4. 如果参考文本不足以回答问题，请说明

返回JSON格式:
{{
    "relevance_score": <0-1分数，表示参考文本对查询的相关性>,
    "reason": "<为什么给出这个相关性分数的理由>",
    "generated_answer": "<根据参考文本生成的完整回答，用GPT的语言表达>"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 使用更好的模型生成回答
                messages=[
                    {"role": "system", "content": "你是一个智能问答助手，能够根据参考文本生成准确、完整的回答。"},
                    {"role": "user", "content": answer_generation_prompt}
                ],
                temperature=0.1,  # 稍微增加创造性
                max_tokens=400  # 允许更长的回答
            )

            result_text = response.choices[0].message.content.strip()
            relevance_score, reason, generated_answer = self._parse_answer_generation_response(result_text, content)

            return relevance_score, reason, generated_answer

        except Exception as e:
            print(f"LLM回答生成失败，使用备用方法: {e}")
            # 备用方法：简单的基于模板的回答生成
            fallback_answer = self._generate_fallback_answer(query, content)
            return vector_score, "使用备用回答生成", fallback_answer
    
    def _build_evaluation_prompt(self, query: str, content: str, vector_score: float) -> str:
        """LLM評価プロンプトを構築"""
        return f"""
以下のテキスト内容とユーザークエリの関連性を評価してください：

**ユーザークエリ：**
{query}

**候補テキスト内容：**
{content}

**ベクトル類似度スコア：** {vector_score:.4f}

以下の形式で評価結果を返してください：

```json
{{
    "relevance_score": <0.0-1.0の間の関連性スコア>,
    "reason": "<詳細な関連性分析、なぜこのスコアを与えたかの説明>",
    "key_points": ["<キーマッチポイント1>", "<キーマッチポイント2>", "..."],
    "highlighted_content": "<原文から正確に抽出したクエリに最も類似する核心部分、質問に最も直接的に答える最も関連性の高い部分を選択（1-2文程度）、文字の追加や修正は行わない>"
}}
```

評価基準：
1. **直接回答性** (0-0.3): テキストがクエリに直接答えているか
2. **内容関連性** (0-0.3): テキスト内容とクエリトピックの関連度
3. **情報完全性** (0-0.2): 提供される情報が包括的で正確か
4. **意味マッチング度** (0-0.2): キーワードと概念のマッチング度

**重要な注意事項：**
- highlighted_contentは原文から正確にコピーしたテキストフラグメントである必要があります
- 質問に最も直接的に答える最も関連性の高い部分を選択してください（1-2文程度）
- 段落全体や過度に長い内容を含めないでください
- 抽出されたテキストが原文と完全に一致することを確認してください
- クエリキーワードを含む部分を優先的に選択してください

有効なJSON形式で返すことを確認してください。
"""
    
    def _parse_llm_response(self, result_text: str, original_content: str) -> Tuple[float, str, str]:
        """LLM応答を解析"""
        try:
            # JSONブロックを抽出
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                relevance_score = float(parsed.get('relevance_score', 0.0))
                reason = parsed.get('reason', '理由不明')
                highlighted_content = parsed.get('highlighted_content', '')
                if not highlighted_content or highlighted_content.strip() == '':
                    highlighted_content = self._extract_first_sentence_safe(original_content)
                
                # highlighted_contentを検証
                highlighted_content = self._validate_highlighted_content(highlighted_content, original_content)
                
                return relevance_score, reason, highlighted_content
            else:
                raise ValueError("有効なJSONが見つかりません")
                
        except Exception as e:
            print(f"LLM応答解析失敗: {e}")
            return 0.5, f"解析失敗: {str(e)}", self._extract_first_sentence_safe(original_content)
    
    def _validate_highlighted_content(self, highlighted_content: str, original_content: str) -> str:
        """highlighted_contentを検証・修正"""
        clean_highlighted = highlighted_content.replace("**", "").replace("*", "").strip()
        
        if clean_highlighted in original_content:
            if len(clean_highlighted) > 200:
                sentences = clean_highlighted.split('。')
                if len(sentences) > 1:
                    return '。'.join(sentences[:2]) + '。'
                else:
                    return clean_highlighted[:150] + "..."
            return clean_highlighted
        
        # 部分マッチングを試行
        best_match = ""
        best_length = 0
        
        for i in range(len(original_content)):
            for j in range(i + 1, len(original_content) + 1):
                substring = original_content[i:j]
                if substring in clean_highlighted and len(substring) > best_length:
                    best_match = substring
                    best_length = len(substring)
        
        if best_length >= 10:
            return best_match
        
        # フォールバック: 原文の最初の50文字
        return self._extract_first_sentence_safe(original_content)

    def _parse_answer_generation_response(self, result_text: str, original_content: str) -> Tuple[float, str, str]:
        """解析LLM生成的回答响应"""
        try:
            # 提取JSON块
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                parsed = json.loads(json_str)

                relevance_score = float(parsed.get('relevance_score', 0.0))
                reason = parsed.get('reason', '未提供理由')
                generated_answer = parsed.get('generated_answer', '')

                if not generated_answer or generated_answer.strip() == '':
                    generated_answer = self._generate_fallback_answer("用户查询", original_content)

                return relevance_score, reason, generated_answer
            else:
                raise ValueError("未找到有效的JSON格式")

        except Exception as e:
            print(f"LLM回答解析失败: {e}")
            fallback_answer = self._generate_fallback_answer("用户查询", original_content)
            return 0.5, f"解析失败: {str(e)}", fallback_answer

    def _generate_fallback_answer(self, query: str, content: str) -> str:
        """生成备用回答 - 当LLM失败时使用"""
        import re

        # 简单的基于关键词匹配的回答生成
        if "種類" in query or "分類" in query:
            # 对于分类查询，尝试提取分类信息
            if "2種類" in content or "二種類" in content:
                return "根据文本内容，主要分为两种类型。"
            elif "普通型" in content and "自立型" in content:
                return "根据文本内容，主要分为普通型和自立型两种类型。"
        elif "とは" in query or "何ですか" in query:
            # 对于定义查询，提取第一句作为定义
            sentences = re.split(r'[。！？.!?]', content)
            if sentences and sentences[0].strip():
                return sentences[0].strip() + "。"

        # 默认返回内容摘要
        return content[:100] + ("..." if len(content) > 100 else "")

    def _extract_first_sentence_safe(self, content: str) -> str:
        """
        简练的关键词片段提取，优先提取最相关的核心关键词
        """
        if not content or content.strip() == '':
            return ""
        
        content = content.strip()
        
        # 关键词分隔符，按重要性排序
        keyword_separators = [
            "、",          # 顿号 (列举分隔)
            "と",          # 和
            "や",          # 和、或
            "の",          # 的
            " ",           # 空格
            "・",          # 中点
            "及び",        # 以及
        ]
        
        # 1. 优先提取关键词片段（不需要完整句子）
        # 寻找最短有意义的片段
        min_length = 8   # 最小有意义长度
        max_length = 50  # 最大长度限制，保持简练
        
        if len(content) <= max_length:
            return content
        
        # 2. 在max_length范围内寻找最佳截断点
        best_cut_point = min_length
        
        # 优先在分隔符处截断
        for i in range(max_length - 1, min_length - 1, -1):
            if i < len(content):
                # 检查是否在关键词分隔符处
                if content[i] in ['、', '・', ' ', 'と', 'や']:
                    best_cut_point = i + 1
                    break
                # 检查是否在自然词汇边界
                if i > 0 and (content[i-1] in ['の', 'に', 'で', 'を', 'が', 'は']):
                    best_cut_point = i
                    break
        
        # 3. 如果找不到好的截断点，在句子边界截断
        sentence_endings = ['。', '！', '？', '.', '!', '?']
        for i in range(max_length - 1, min_length - 1, -1):
            if i < len(content) and content[i] in sentence_endings:
                return content[:i + 1]
        
        # 4. 最后resort：简单截断到max_length
        if best_cut_point < len(content):
            result = content[:best_cut_point]
            # 避免在汉字/假名中间截断
            if len(result) > 0 and ord(result[-1]) > 127:  # 非ASCII字符
                return result
            else:
                return result + "..."
        
        return content[:max_length] + "..." if len(content) > max_length else content


class UltraFastRAG:
    """統合された超高速RAGシステム"""
    
    def __init__(self, openai_api_key: str, chroma_path: str = "./chroma", use_llm_ranking: bool = True, highlight_mode: str = "auto"):
        self.openai_api_key = openai_api_key
        self.chroma_path = chroma_path
        self.use_llm_ranking = use_llm_ranking
        self.highlight_mode = highlight_mode  # "auto", "sentence", "keyword"
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        
        # 向量データベースを初期化
        self.db = None
        if os.path.exists(chroma_path):
            try:
                self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
                print(f"✅ 既存の向量データベースを読み込み: {chroma_path}")
            except Exception as e:
                print(f"⚠️ 向量データベース読み込み失敗: {e}")
        
        # LLMランキングシステムを初期化
        if use_llm_ranking:
            try:
                self.llm_ranker = IntegratedLLMEvidenceRanker(openai_api_key, highlight_mode=highlight_mode)
                print(f"✅ LLM智能ranking已启用 (高亮模式: {highlight_mode})")
            except Exception as e:
                print(f"⚠️ LLM ranking初始化失敗: {e}，将使用原始方法")
                self.use_llm_ranking = False
        else:
            self.use_llm_ranking = False
    
    def build_multi_granular_vector_store(self, data_file: str) -> bool:
        """多粒度向量数据库构建 - 句子、短段落、长段落三种粒度"""
        try:
            print(f"🗃️ 构建多粒度向量数据库...")
            print(f"📁 数据文件: {data_file}")
            print(f"🗄️ 向量数据库路径: {self.chroma_path}")
            
            # 检查数据文件
            if not os.path.exists(data_file):
                print(f"❌ 数据文件不存在: {data_file}")
                return False
            
            # 加载数据
            print("📖 加载数据...")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 加载了 {len(data)} 条数据")
            
            # 转换为Document格式
            documents = []
            for item in data:
                content = item.get('output', '') or item.get('text', '') or item.get('content', '')
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': 'multi_granular',
                            'original_index': str(len(documents))
                        }
                    )
                    documents.append(doc)
            
            print(f"📄 转换了 {len(documents)} 个文档")
            
            # 多粒度分块
            print("🔀 执行多粒度分块...")
            all_chunks = self._create_multi_granular_chunks(documents)
            
            print(f"📊 多粒度分块统计:")
            for granularity, chunks in all_chunks.items():
                avg_size = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
                print(f"  {granularity}: {len(chunks)} chunks (平均 {avg_size:.1f} 字符)")
            
            # 合并所有粒度的chunks
            all_combined_chunks = []
            for granularity, chunks in all_chunks.items():
                all_combined_chunks.extend(chunks)
            
            print(f"📦 总计 {len(all_combined_chunks)} 个多粒度chunks")
            
            # 清理旧的向量库
            if os.path.exists(self.chroma_path):
                import shutil
                shutil.rmtree(self.chroma_path)
                print("🗑️ 清理旧向量库")
            
            # 创建向量库
            print("🔄 创建多粒度向量库...")
            start_time = time.time()
            
            self.db = Chroma.from_documents(
                all_combined_chunks,
                self.embedding_function,
                persist_directory=self.chroma_path
            )
            
            build_time = time.time() - start_time
            print(f"✅ 多粒度向量库构建完成! 耗时: {build_time:.2f}s")
            print(f"📊 统计: {len(all_combined_chunks)} chunks, 平均 {build_time/len(all_combined_chunks)*1000:.1f}ms/chunk")
            
            return True
            
        except Exception as e:
            print(f"❌ 多粒度向量库构建失败: {e}")
            return False
    
    def _create_multi_granular_chunks(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """创建多粒度chunks: 句子级、短段落级、长段落级"""
        import re
        
        all_chunks = {
            'sentence': [],      # 句子级 (10-60字符)
            'short_passage': [], # 短段落级 (80-200字符) 
            'long_passage': []   # 长段落级 (300-500字符)
        }
        
        for doc_idx, doc in enumerate(documents):
            content = doc.page_content.strip()
            if not content:
                continue
                
            # 1. 句子级分块
            sentence_chunks = self._create_sentence_chunks(content, doc_idx)
            all_chunks['sentence'].extend(sentence_chunks)
            
            # 2. 短段落级分块
            short_chunks = self._create_short_passage_chunks(content, doc_idx)
            all_chunks['short_passage'].extend(short_chunks)
            
            # 3. 长段落级分块
            long_chunks = self._create_long_passage_chunks(content, doc_idx)
            all_chunks['long_passage'].extend(long_chunks)
        
        return all_chunks
    
    def _create_sentence_chunks(self, content: str, doc_idx: int) -> List[Document]:
        """创建句子级chunks (10-60字符)"""
        import re
        
        # 句子分割
        sentences = re.split(r'[。！？.!?]', content)
        chunks = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if 10 <= len(sentence) <= 60 and sentence:
                chunk_doc = Document(
                    page_content=sentence,
                    metadata={
                        'granularity': 'sentence',
                        'doc_index': doc_idx,
                        'chunk_index': i,
                        'chunk_size': len(sentence),
                        'chunk_type': 'sentence'
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks
    
    def _create_short_passage_chunks(self, content: str, doc_idx: int) -> List[Document]:
        """创建短段落级chunks (80-200字符)"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=30,
            length_function=len,
            separators=["\\n\\n", "。", "！", "？", "、", "\\n", " ", ""]
        )
        
        chunks = []
        raw_chunks = splitter.split_text(content)
        
        for i, chunk_text in enumerate(raw_chunks):
            if 80 <= len(chunk_text) <= 200:
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        'granularity': 'short_passage',
                        'doc_index': doc_idx,
                        'chunk_index': i,
                        'chunk_size': len(chunk_text),
                        'chunk_type': 'short_passage'
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks
    
    def _create_long_passage_chunks(self, content: str, doc_idx: int) -> List[Document]:
        """创建长段落级chunks (300-500字符)"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            length_function=len,
            separators=["\\n\\n", "。", "！", "？", "\\n", " ", ""]
        )
        
        chunks = []
        raw_chunks = splitter.split_text(content)
        
        for i, chunk_text in enumerate(raw_chunks):
            if 300 <= len(chunk_text) <= 500:
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        'granularity': 'long_passage',
                        'doc_index': doc_idx,
                        'chunk_index': i,
                        'chunk_size': len(chunk_text),
                        'chunk_type': 'long_passage'
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks
    
    def _multi_granular_retrieval(self, query: str, k: int = 10) -> List[Dict]:
        """多粒度检索 - 自动选择最小充分单元"""
        if not self.db:
            return []
        
        print(f"🔍 执行多粒度检索: {query}")
        
        # 1. 获取所有粒度的候选结果
        all_results = self.db.similarity_search_with_score(query, k=k*3)  # 获取更多候选
        
        # 2. 按粒度分组
        granularity_groups = {
            'sentence': [],
            'short_passage': [],
            'long_passage': []
        }
        
        for doc, score in all_results:
            granularity = doc.metadata.get('granularity', 'unknown')
            if granularity in granularity_groups:
                granularity_groups[granularity].append({
                    'document': doc,
                    'score': score,
                    'granularity': granularity,
                    'chunk_size': doc.metadata.get('chunk_size', len(doc.page_content))
                })
        
        # 3. logits引导的多粒度选择算法
        selected_chunks = self._logits_guided_selection(granularity_groups, query, k)
        
        print(f"📊 多粒度检索结果:")
        for granularity, count in self._count_by_granularity(selected_chunks).items():
            print(f"  {granularity}: {count} chunks")
        
        return selected_chunks
    
    def _logits_guided_selection(self, granularity_groups: Dict, query: str, k: int) -> List[Dict]:
        """Logits引导的多粒度选择算法"""
        
        # 1. 计算查询复杂度
        query_complexity = self._calculate_query_complexity(query)
        print(f"🎯 查询复杂度: {query_complexity}")
        
        # 2. 基于复杂度确定粒度偏好
        if query_complexity <= 0.3:
            # 简单查询 - 偏好句子级
            granularity_weights = {'sentence': 1.0, 'short_passage': 0.5, 'long_passage': 0.2}
            print("📝 简单查询 - 优先句子级别")
        elif query_complexity <= 0.7:
            # 中等查询 - 偏好短段落级
            granularity_weights = {'sentence': 0.6, 'short_passage': 1.0, 'long_passage': 0.7}
            print("📄 中等查询 - 优先短段落级别")
        else:
            # 复杂查询 - 偏好长段落级
            granularity_weights = {'sentence': 0.3, 'short_passage': 0.8, 'long_passage': 1.0}
            print("📚 复杂查询 - 优先长段落级别")
        
        # 3. 重新计算加权分数
        all_weighted_chunks = []
        for granularity, chunks in granularity_groups.items():
            weight = granularity_weights.get(granularity, 0.5)
            for chunk in chunks:
                weighted_score = chunk['score'] * weight
                chunk['weighted_score'] = weighted_score
                all_weighted_chunks.append(chunk)
        
        # 4. 按加权分数排序并选择top-k
        all_weighted_chunks.sort(key=lambda x: x['weighted_score'])
        selected = all_weighted_chunks[:k]
        
        # 5. 最小充分单元检查
        return self._ensure_minimal_sufficient_units(selected, query)
    
    def _calculate_query_complexity(self, query: str) -> float:
        """计算查询复杂度 (0-1之间)"""
        complexity_score = 0.0
        
        # 长度因子
        if len(query) > 20:
            complexity_score += 0.3
        elif len(query) > 10:
            complexity_score += 0.1
        
        # 复杂词汇因子
        complex_patterns = ['について', 'とは何', 'どのように', '違いは', '特徴', '方法', '理由']
        for pattern in complex_patterns:
            if pattern in query:
                complexity_score += 0.2
                break
        
        # 多概念因子
        import re
        concepts = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', query)
        if len(concepts) > 3:
            complexity_score += 0.3
        elif len(concepts) > 1:
            complexity_score += 0.1
        
        return min(complexity_score, 1.0)
    
    def _ensure_minimal_sufficient_units(self, chunks: List[Dict], query: str) -> List[Dict]:
        """确保选择最小充分单元"""
        if not chunks:
            return chunks
        
        # 按粒度分组已选择的chunks
        selected_by_granularity = {}
        for chunk in chunks:
            granularity = chunk['granularity']
            if granularity not in selected_by_granularity:
                selected_by_granularity[granularity] = []
            selected_by_granularity[granularity].append(chunk)
        
        # 如果有句子级别且相关性高，优先使用
        if 'sentence' in selected_by_granularity:
            sentence_chunks = selected_by_granularity['sentence']
            high_quality_sentences = [c for c in sentence_chunks if c['weighted_score'] < 0.3]  # 低分数=高相关性
            if high_quality_sentences:
                print("✅ 选择句子级别作为最小充分单元")
                return high_quality_sentences[:3]  # 最多3个句子
        
        # 否则混合选择
        print("🔄 使用混合粒度策略")
        return chunks[:5]  # 最多5个混合粒度
    
    def _count_by_granularity(self, chunks: List[Dict]) -> Dict[str, int]:
        """统计各粒度的chunk数量"""
        counts = {'sentence': 0, 'short_passage': 0, 'long_passage': 0}
        for chunk in chunks:
            granularity = chunk['granularity']
            if granularity in counts:
                counts[granularity] += 1
        return counts
    
    def query_with_multi_granular(self, query: str, k: int = 5) -> Tuple[str, str, str, int, int]:
        """多粒度查询 - 使用多粒度检索"""
        if not self.db:
            return "向量数据库未初始化", "", "", 0, 0
        
        print(f"🔍 执行多粒度检索: {query}")
        
        # 1. 获取所有粒度的候选结果
        all_results = self.db.similarity_search_with_score(query, k=k*3)  # 获取更多候选
        
        # 2. 按粒度分组
        granularity_groups = {
            'sentence': [],
            'short_passage': [],
            'long_passage': []
        }
        
        for doc, score in all_results:
            granularity = doc.metadata.get('granularity', 'unknown')
            if granularity in granularity_groups:
                granularity_groups[granularity].append({
                    'document': doc,
                    'score': score,
                    'granularity': granularity,
                    'chunk_size': doc.metadata.get('chunk_size', len(doc.page_content))
                })
        
        # 3. logits引导的多粒度选择算法
        selected_chunks = self._logits_guided_selection(granularity_groups, query, k)
        
        print(f"📊 多粒度检索结果:")
        for granularity, count in self._count_by_granularity(selected_chunks).items():
            print(f"  {granularity}: {count} chunks")
        
        # 4. 选择最佳匹配
        if selected_chunks:
            best_chunk = selected_chunks[0]
            doc = best_chunk['document']
            source_text = doc.page_content
            evidence_text = source_text  # 简化处理
            answer = self._generate_answer_fast(evidence_text, query)
            return answer, source_text, evidence_text, 0, len(evidence_text)
        else:
            return "未找到相关信息", "", "", 0, 0

    def build_vector_store(self, data_file: str) -> bool:
        """精細chunking を使用して向量データベースを構築"""
        try:
            print(f"🏗️ 精細chunking で向量データベース構築中...")
            print(f"📁 データファイル: {data_file}")
            print(f"🗄️ 向量データベースパス: {self.chroma_path}")
            
            if not os.path.exists(data_file):
                print(f"❌ データファイルが見つかりません: {data_file}")
                return False
            
            # データを読み込み
            print("📖 データ読み込み中...")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ {len(data)} 件のデータを読み込み")
            
            # Documentに変換
            documents = []
            for item in data:
                content = item.get('output', '') or item.get('text', '') or item.get('content', '')
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': 'dataset',
                        'index': str(len(documents))
                    }
                )
                documents.append(doc)
            
            print(f"📄 {len(documents)} 個のドキュメントに変換")
            
            # 精細文本分割
            print("✂️ 精細chunking実行中...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=150,        # 150字符 (約1-2文)
                chunk_overlap=30,      # 30字符重複
                length_function=len,
                add_start_index=True,
                separators=[
                    "\n\n",           # 段落
                    "。",             # 句号  
                    "！",             # 感嘆号
                    "？",             # 疑問符
                    "、",             # 読点
                    "\n",             # 改行
                    " ",              # 空格
                    ""                # 字符级别
                ]
            )
            chunks = text_splitter.split_documents(documents)
            avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks)
            print(f"🔪 {len(chunks)} 個のchunksに分割 (平均 {avg_chunk_size:.1f} 文字/chunk)")
            
            # 既存の向量データベースを削除
            if os.path.exists(self.chroma_path):
                import shutil
                shutil.rmtree(self.chroma_path)
                print("🗑️ 既存の向量データベースを削除")
            
            # 新しい向量データベースを作成
            print("🔄 向量データベース作成中...")
            start_time = time.time()
            
            self.db = Chroma.from_documents(
                chunks,
                self.embedding_function,
                persist_directory=self.chroma_path
            )
            
            build_time = time.time() - start_time
            print(f"✅ 向量データベース構築完了! 時間: {build_time:.2f}s")
            print(f"📊 統計: {len(chunks)} chunks, 平均 {build_time/len(chunks)*1000:.1f}ms/chunk")
            
            return True
            
        except Exception as e:
            print(f"❌ 向量データベース構築失敗: {e}")
            return False
    
    def _smart_sentence_split(self, text: str) -> List[str]:
        """智能句子分割"""
        import re
        
        # 句子结束标记
        sentence_endings = r'[。！？.!?;；]'
        
        # 分割句子
        sentences = re.split(sentence_endings, text)
        
        # 清理并过滤
        clean_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) >= 5:  # 至少5个字符
                # 确保句子完整性
                if not sent.endswith(('。', '！', '？', '.', '!', '?')):
                    sent += '。'
                clean_sentences.append(sent)
        
        return clean_sentences
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询关键词 - 改进版"""
        import re
        
        # 分离日语助词和实质词汇
        # 先移除常见助词和疑问词
        clean_query = query
        for remove_word in ['とは何', 'とは', '何ですか', '何でしょうか', 'について', 'ですか', 'でしょうか', 'です', 'ます']:
            clean_query = clean_query.replace(remove_word, ' ')
        
        # 提取有意义的词汇
        keywords = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+', clean_query)
        
        # 过滤停用词和太短的词
        stopwords = {'何', 'の', 'を', 'に', 'が', 'は', 'で', 'と', 'から', 'まで', 'より', 'ほど', '的', '了', '在', '是', '有', '这', '那', '一'}
        keywords = [kw.strip() for kw in keywords if kw.strip() and len(kw.strip()) > 1 and kw.strip() not in stopwords]
        
        # 去重并保持顺序
        unique_keywords = []
        for kw in keywords:
            if kw not in unique_keywords:
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _calculate_keyword_match_score(self, text: str, keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matched_count = 0
        total_matches = 0
        
        for keyword in keywords:
            count = text.count(keyword)
            if count > 0:
                matched_count += 1
                total_matches += count
        
        # 匹配度评分
        coverage_score = matched_count / len(keywords)  # 关键词覆盖率
        density_score = min(total_matches / len(text) * 100, 1.0)  # 密度分数
        
        return (coverage_score * 0.7) + (density_score * 0.3)
    
    def _query_with_keyword_boost(self, query_text: str, k: int = 10) -> Tuple[str, str, str, int, int]:
        """增强关键词匹配的查询方法"""
        print(f"🔍 关键词增强查询: '{query_text}'")
        
        # 1. 向量检索 (扩大搜索范围)
        search_results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
        
        if not search_results:
            return "信息未找到。", "", "", 0, 0
        
        # 2. 关键词精确匹配增强
        enhanced_results = []
        keywords = self._extract_keywords(query_text)
        print(f"🔑 提取关键词: {keywords}")
        
        for doc, vector_score in search_results:
            keyword_score = self._calculate_keyword_match_score(doc.page_content, keywords)
            
            # 综合评分：关键词匹配权重更高
            final_score = (vector_score * 0.3) + (keyword_score * 0.7)
            
            enhanced_results.append((doc, final_score, keyword_score, vector_score))
        
        # 按综合评分排序
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最佳匹配
        best_doc, best_score, kw_score, vec_score = enhanced_results[0]
        
        source_text = best_doc.page_content
        evidence_text = source_text  # 句子级别，直接使用
        
        # 生成答案
        answer = self._generate_answer_fast(evidence_text, query_text)
        
        print(f"🎯 最佳匹配: 综合评分={best_score:.3f}, 向量={vec_score:.3f}, 关键词={kw_score:.3f}")
        
        return answer, source_text, evidence_text, 0, len(evidence_text)

    def query(self, query_text: str, k: int = 5, use_keyword_boost: bool = False) -> Tuple[str, str, str, int, int]:
        """クエリ処理 - 多种模式支持"""
        if not self.db:
            return "向量データベースが初期化されていません。", "", "", 0, 0
        
        if use_keyword_boost:
            # 使用关键词增强模式 (适合精确匹配)
            return self._query_with_keyword_boost(query_text, k)
        elif self.use_llm_ranking:
            # 使用LLM智能排序模式 (适合复杂语义理解)
            return self._query_with_llm_ranking(query_text, k)
        else:
            # 使用原始快速模式
            return self._query_fast_original(query_text)
    
    def _query_with_llm_ranking(self, query_text: str, k: int) -> Tuple[str, str, str, int, int]:
        """LLM ranking を使用した拡張クエリ"""
        print(f"🧠 LLM智能ranking でクエリ処理中: '{query_text}'")
        
        # 向量類似検索
        search_results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
        
        if not search_results:
            return "情報が見つかりませんでした。", "", "", 0, 0
        
        # LLM ranker 形式に変換
        chunks = []
        for doc, score in search_results:
            chunk = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            }
            chunks.append(chunk)
        
        # LLM智能排序
        try:
            ranked_chunks = self.llm_ranker.rank_and_highlight_chunks(query_text, chunks, top_k=1)
            
            if ranked_chunks:
                best_chunk = ranked_chunks[0]
                source_text = best_chunk["content"]
                evidence_text = best_chunk.get("highlighted_content", source_text)
                
                start_pos = 0
                end_pos = len(evidence_text)
                
                # 回答生成
                answer = self._generate_answer_fast(evidence_text, query_text)
                
                print(f"✅ LLM ranking完成，最佳匹配スコア: {best_chunk.get('final_score', 0):.3f}")
                
                # 🚨 终极安全检查 - 绝对不返回整段文本
                if len(evidence_text) > 120 or len(evidence_text) >= len(source_text) * 0.7:
                    print(f"🚨 最终强制修复: {len(evidence_text)} -> ", end="")
                    evidence_text = self.llm_ranker.extract_single_sentence_forced(source_text)
                    start_pos = source_text.find(evidence_text) if evidence_text in source_text else 0
                    end_pos = start_pos + len(evidence_text)
                    print(f"{len(evidence_text)}字符")
                
                # 双重检查
                if len(evidence_text) > 120:
                    print(f"🚨 双重检查触发: evidence仍然过长({len(evidence_text)}字符)")
                    evidence_text = self.llm_ranker.extract_single_sentence_forced(source_text)
                    start_pos = 0
                    end_pos = len(evidence_text)
                
                return answer, source_text, evidence_text, start_pos, end_pos
            
        except Exception as e:
            print(f"⚠️ LLM ranking失敗: {e}，原始方法にフォールバック")
        
        # 原始方法にフォールバック
        return self._query_fast_original(query_text)
    
    def _query_fast_original(self, query_text: str) -> Tuple[str, str, str, int, int]:
        """原始の超高速クエリ方法"""
        search_results = self.db.similarity_search_with_relevance_scores(query_text, k=1)
        
        if not search_results:
            return "情報が見つかりませんでした。", "", "", 0, 0
        
        hit_doc = search_results[0][0]
        confidence = search_results[0][1]
        
        source_text = hit_doc.page_content
        evidence_text, start_pos, end_pos = self._extract_evidence_fast(source_text, query_text)
        answer = self._generate_answer_fast(evidence_text, query_text)
        
        return answer, source_text, evidence_text, start_pos, end_pos
    
    def _extract_evidence_fast(self, text: str, query: str) -> Tuple[str, int, int]:
        """高速証拠抽出"""
        keywords = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+', query)
        keywords = [kw for kw in keywords if len(kw) > 1 and kw not in ['とは', '何', 'です', 'ます', 'について']]
        
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s for s in sentences if s]
        
        if not sentences:
            snippet = text[:100]
            return snippet, 0, len(snippet)
        
        best_sentence = None
        best_score = -1
        
        for sentence in sentences:
            score = 0
            for kw in keywords:
                if kw in sentence:
                    score += 1
            
            if score > best_score:
                best_sentence = sentence
                best_score = score
        
        if best_sentence is None:
            best_sentence = sentences[0]
        
        start_pos = text.find(best_sentence)
        if start_pos < 0:
            best_sentence = sentences[0]
            start_pos = 0
        end_pos = start_pos + len(best_sentence)
        return best_sentence.strip(), start_pos, end_pos
    
    def _generate_answer_fast(self, evidence: str, query: str) -> str:
        """高速回答生成"""
        if any(pattern in query for pattern in ['とは何', 'とは', '何ですか', '何でしょうか']):
            first_sentence = re.split(r'[。！？.!?]', evidence)[0]
            if first_sentence:
                return first_sentence + '。'
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "簡潔に日本語で答えてください。"},
                    {"role": "user", "content": f"証拠: {evidence}\n質問: {query}\n回答:"}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except:
            return evidence[:100] + ('...' if len(evidence) > 100 else '')


def test_integrated_system():
    """統合システムのテスト"""
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未設定")
        return
    
    print("🚀 統合RAGシステム テスト")
    print("=" * 50)
    
    # システム初期化
    rag = UltraFastRAG(api_key, "./chroma_integrated", use_llm_ranking=True)
    
    # データベース構築をテスト (オプション)
    # rag.build_vector_store("../data/single_20240229.json")
    
    # クエリテスト
    queries = [
        "コンバインとは何ですか",
        "音位転倒について説明してください",
        "漢方薬の違いは何ですか"
    ]
    
    for query in queries:
        print(f"\n🔍 クエリ: {query}")
        
        start_time = time.time()
        answer, source, evidence, start, end = rag.query(query)
        elapsed = time.time() - start_time
        
        print(f"⏱️  処理時間: {elapsed:.2f}秒")
        print(f"💬 回答: {answer}")
        print(f"🔍 証拠: {evidence}")
        print(f"📊 証拠範囲: {start+1}〜{end}文字目")
        print("-" * 40)


def test_improved_rag():
    """改进版RAG测试 - 对比不同模式"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return
    
    print("🚀 改进版RAG系统测试 - 多模式对比")
    print("=" * 60)
    
    # 初始化系统
    rag = UltraFastRAG(api_key, "./chroma_sentence_level", use_llm_ranking=True)
    
    # 可选：构建句子级别数据库
    # print("🗃️ 构建句子级别向量数据库...")
    # rag.build_vector_store_improved("../data/single_20240229.json", use_sentence_level=True)
    
    # 测试查询
    queries = [
        "コンバインとは何ですか",
        "機械学習とは何ですか", 
        "日本の農業について"
    ]
    
    modes = [
        ("🔑 关键词增强模式", {"use_keyword_boost": True}),
        ("🧠 LLM智能模式", {"use_keyword_boost": False}),
        ("⚡ 快速模式", {"use_keyword_boost": False})
    ]
    
    for query in queries:
        print(f"\n{'='*20} 查询: {query} {'='*20}")
        
        for mode_name, params in modes:
            print(f"\n{mode_name}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                if mode_name == "⚡ 快速模式":
                    # 临时禁用LLM ranking进行快速模式测试
                    original_llm_setting = rag.use_llm_ranking
                    rag.use_llm_ranking = False
                    answer, source, evidence, start, end = rag.query(query, k=5, **params)
                    rag.use_llm_ranking = original_llm_setting
                else:
                    answer, source, evidence, start, end = rag.query(query, k=5, **params)
                elapsed = time.time() - start_time
                
                print(f"⏱️  处理时间: {elapsed:.2f}秒")
                print(f"💬 回答: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                print(f"🔍 证据: {evidence[:80]}{'...' if len(evidence) > 80 else ''}")
                print(f"📊 证据长度: {len(evidence)}字符")
                
            except Exception as e:
                print(f"❌ 错误: {e}")
        
        print("=" * 80)


def test_highlighting_modes():
    """测试不同高亮模式的效果"""
    print("🔦 测试不同高亮模式")
    print("=" * 60)
    
    # 加载环境变量
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return
    
    # 测试查询
    test_query = "コンバインとは何ですか"
    
    # 测试不同模式
    modes = [
        ("auto", "自动模式"),
        ("sentence", "句子模式"),
        ("keyword", "关键词模式")
    ]
    
    for mode, mode_name in modes:
        print(f"\n📋 测试 {mode_name} ({mode})")
        print("-" * 40)
        
        try:
            rag = UltraFastRAG(
                openai_api_key=api_key,
                chroma_path="./chroma",
                use_llm_ranking=True,
                highlight_mode=mode
            )
            
            start_time = time.time()
            answer, source, evidence, start, end = rag.query(test_query, k=5)
            elapsed = time.time() - start_time
            
            print(f"⏱️  处理时间: {elapsed:.2f}秒")
            print(f"💬 回答: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"🔍 证据: {evidence}")
            print(f"📊 证据长度: {len(evidence)}字符")
            print(f"📄 源文本长度: {len(source)}字符")
            print(f"📈 压缩比: {len(evidence)/len(source)*100:.1f}%")
            
        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == "__main__":
    test_highlighting_modes()