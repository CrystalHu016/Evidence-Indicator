# SmartNews Business Enhanced Crawler with LLM Integration

基于SmartNews Business网站知识的智能回答系统

## 🎯 系统概述

这是一个增强的爬虫系统，专门用于从 [SmartNews Business](https://business.smartnews.com/) 网站收集信息，并使用LLM（大语言模型）生成智能回答，创建结构化的数据集。

## ✨ 主要特性

- **智能内容抓取**: 自动抓取SmartNews Business网站的关键页面
- **LLM集成**: 使用OpenAI GPT模型基于网站内容生成回答
- **多语言支持**: 支持英文和中文查询
- **结构化输出**: 生成标准化的JSON数据集
- **配置化管理**: 灵活的配置文件支持自定义设置
- **错误处理**:  robust的错误处理和重试机制

## 🏗️ 系统架构

```
enhanced_crawler.py      # 主要爬虫类
├── crawler_config.py    # 配置文件
├── run_enhanced_crawler.py  # 运行脚本
└── smartnews_dataset/   # 输出数据集目录
```

## 📋 数据集格式

生成的每个数据集条目包含以下信息：

```json
{
  "core_query": "What is SmartNews and what is their mission?",
  "answer": "基于网站内容生成的LLM回答...",
  "original_urls": [
    "https://business.smartnews.com",
    "https://business.smartnews.com/newsroom",
    "https://business.smartnews.com/newsroom/blogs"
  ],
  "content_summary": "内容来源摘要",
  "timestamp": "2024-01-01T12:00:00",
  "source": "SmartNews Business",
  "crawler_version": "enhanced_v1.0",
  "metadata": {
    "word_count": 150,
    "content_sections": ["main_page", "newsroom", "blogs"],
    "publish_date": null,
    "article_type": "company_info"
  }
}
```

## 🚀 快速开始

### 1. 环境准备

确保你的系统已安装Python 3.8+和必要的依赖包：

```bash
pip install requests beautifulsoup4 openai python-dotenv
```

### 2. 配置OpenAI API

在项目根目录创建 `.env` 文件：

```bash
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. 运行爬虫

```bash
cd script
python run_enhanced_crawler.py
```

## ⚙️ 配置选项

### 核心查询配置

在 `crawler_config.py` 中可以自定义核心查询：

```python
CORE_QUERIES = [
    "What is SmartNews and what is their mission?",
    "What are SmartNews' company values?",
    # 添加更多查询...
]

CORE_QUERIES_ZH = [
    "SmartNews是什么公司，他们的使命是什么？",
    "SmartNews的公司价值观是什么？",
    # 添加更多中文查询...
]
```

### 爬取配置

```python
REQUEST_CONFIG = {
    'timeout': 30,                    # 请求超时时间
    'delay_between_requests': 2.0,    # 请求间隔延迟
    'max_retries': 3,                 # 最大重试次数
}
```

### LLM配置

```python
LLM_CONFIG = {
    'model': 'gpt-3.5-turbo',        # 使用的模型
    'max_tokens': 500,                # 最大输出token数
    'temperature': 0.3,               # 创造性控制
}
```

## 🔧 自定义配置

### 添加新的网站页面

在 `crawler_config.py` 中添加新的URL：

```python
BASE_URLS = {
    'main': 'https://business.smartnews.com',
    'newsroom': 'https://business.smartnews.com/newsroom',
    'blogs': 'https://business.smartnews.com/newsroom/blogs',
    'company': 'https://business.smartnews.com/company',
    'careers': 'https://business.smartnews.com/careers',
    'publishers': 'https://business.smartnews.com/publishers',
    'new_section': 'https://business.smartnews.com/new_section'  # 新增
}
```

### 自定义内容解析

在 `enhanced_crawler.py` 中添加新的解析方法：

```python
def parse_new_section_content(self, html: str) -> List[Dict[str, str]]:
    """解析新页面内容"""
    # 实现自定义解析逻辑
    pass
```

## 📊 输出管理

### 数据集存储

- **默认目录**: `smartnews_dataset/`
- **文件格式**: JSON
- **命名规则**: `smartnews_dataset_YYYYMMDD_HHMMSS.json`

### 数据验证

系统会自动验证生成的数据集：

- 检查所有必需字段
- 验证URL有效性
- 确保内容完整性

## 🛠️ 故障排除

### 常见问题

1. **OpenAI API错误**
   - 检查API key是否正确设置
   - 确认API配额是否充足

2. **网络连接问题**
   - 检查网络连接
   - 调整请求延迟设置

3. **内容解析失败**
   - 检查网站结构是否发生变化
   - 更新相应的CSS选择器

### 日志查看

系统会生成详细的日志信息：

```bash
tail -f script/crawler.log
```

## 🔮 扩展功能

### 集成到RAG系统

生成的数据集可以直接用于你的RAG系统：

```python
from enhanced_crawler import SmartNewsBusinessCrawler

# 创建爬虫实例
crawler = SmartNewsBusinessCrawler(api_key)

# 生成数据集
dataset = crawler.crawl_and_generate_dataset()

# 保存数据集
crawler.save_dataset(dataset, "smartnews_rag_dataset.json")
```

### 支持更多网站

系统设计为可扩展的，可以轻松添加对其他网站的支持：

```python
class OtherWebsiteCrawler(SmartNewsBusinessCrawler):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.base_url = 'https://other-website.com'
        # 自定义配置...
```

## 📝 使用示例

### 基本使用

```python
from enhanced_crawler import SmartNewsBusinessCrawler

# 初始化爬虫
crawler = SmartNewsBusinessCrawler(api_key="your_api_key")

# 运行完整流程
result = crawler.run_full_pipeline()

if result:
    print(f"数据集已保存到: {result}")
```

### 自定义查询

```python
# 自定义查询
custom_queries = [
    "How does SmartNews handle user privacy?",
    "What are the latest business developments?"
]

# 为自定义查询生成回答
for query in custom_queries:
    answer = crawler.generate_llm_answer(query, context_content)
    print(f"问题: {query}")
    print(f"回答: {answer}")
```

## 📄 许可证

本项目遵循MIT许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个系统！

---

**🎯 开始使用SmartNews Business增强爬虫系统，创建你的智能数据集！**
