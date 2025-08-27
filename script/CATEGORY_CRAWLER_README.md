# Yahoo!ニュース分类新闻爬虫

这个爬虫专门用于爬取Yahoo!ニュース每个类别下的最新新闻，并生成日语查询和答案。

## 功能特点

- 🎯 **分类爬取**: 爬取10个主要新闻类别的新闻
- 📰 **最新新闻**: 每个类别获取最新的2条新闻
- 🤖 **LLM集成**: 使用OpenAI API生成智能查询和答案
- 🇯🇵 **日语支持**: 查询和答案都使用日语
- 💾 **数据记录**: 记录timestamp、original_urls、category等信息

## 支持的新闻类别

1. **主要** - 主要新闻
2. **国内** - 国内新闻
3. **国際** - 国际新闻
4. **経済** - 经济新闻
5. **エンタメ** - 娱乐新闻
6. **スポーツ** - 体育新闻
7. **IT** - IT技术新闻
8. **科学** - 科学新闻
9. **ライフ** - 生活新闻
10. **地域** - 地区新闻

## 安装依赖

```bash
pip install requests beautifulsoup4 openai python-dotenv
```

## 环境变量设置

创建`.env`文件并设置：

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 使用方法

### 方法1: 直接运行爬虫

```bash
python category_news_crawler.py
```

### 方法2: 使用运行脚本

```bash
python run_category_crawler.py
```

### 方法3: 在代码中使用

```python
from category_news_crawler import CategoryNewsCrawler

# 创建爬虫实例
crawler = CategoryNewsCrawler(openai_api_key="your_key")

# 运行爬虫
news_data = crawler.crawl_category_news()

# 保存数据集
crawler.save_dataset(news_data, "my_dataset.json")
```

## 输出数据结构

每个新闻条目包含以下信息：

```json
{
  "core_query": "日语查询",
  "answer": "日语答案",
  "original_urls": ["新闻URL"],
  "content_summary": "新闻标题",
  "timestamp": "爬取时间",
  "source": "Yahoo!ニュース",
  "category": "新闻类别",
  "crawler_version": "爬虫版本"
}
```

## 爬虫特性

### 智能内容提取
- 自动识别新闻链接和标题
- 提取新闻详细内容
- 支持多种HTML选择器

### 查询生成
- **LLM模式**: 使用OpenAI API生成智能查询和答案
- **备用模式**: 如果没有API key，使用模板生成简单查询

### 反爬虫保护
- 随机延迟请求
- 模拟真实浏览器请求头
- 错误处理和重试机制

## 注意事项

1. **请求频率**: 爬虫包含延迟机制，避免对服务器造成压力
2. **API限制**: 使用OpenAI API时注意token使用量和速率限制
3. **内容准确性**: 生成的查询和答案基于爬取的新闻内容
4. **法律合规**: 请遵守网站的robots.txt和使用条款

## 故障排除

### 常见问题

1. **无法获取新闻内容**
   - 检查网络连接
   - 确认网站结构是否发生变化
   - 查看日志输出

2. **LLM生成失败**
   - 检查OpenAI API key是否正确
   - 确认API配额是否充足
   - 查看错误日志

3. **爬取速度慢**
   - 调整延迟参数
   - 检查网络延迟
   - 考虑使用代理

### 日志查看

爬虫会输出详细的日志信息，包括：
- 爬取进度
- 错误信息
- 成功统计
- 性能指标

## 扩展功能

### 自定义类别
可以在`category_urls`字典中添加新的新闻类别：

```python
self.category_urls['新类别'] = 'https://news.yahoo.co.jp/categories/new_category'
```

### 自定义选择器
可以修改`extract_news_from_category`方法中的选择器来适应不同的网站结构。

### 数据导出
支持多种数据格式导出，可以根据需要扩展保存方法。

## 许可证

本项目仅供学习和研究使用，请遵守相关法律法规和网站使用条款。
