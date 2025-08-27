## grab data from cross-site in order to generate dataset for Evidence Indicator
import requests
from bs4 import BeautifulSoup
import os

base_url = 'https://business.smartnews.com/newsroom/blogs'
folder_name = 'smartnews_blogs'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def get_page_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    return None

def parse_blog_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    # 这里需要根据实际页面结构修改选择器，例如查找文章列表的容器及其a标签
    articles = soup.select('article a')  # 示例选择器，请替换为网站实际结构
    for a in articles:
        href = a.get('href')
        if href and href.startswith('https://business.smartnews.com/newsroom/blogs/'):
            if href not in links:
                links.append(href)
    return links

def parse_blog_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.find('h1')
    title = title_tag.text.strip() if title_tag else 'no_title'
    content_tags = soup.select('div.blog-content p')  # 示例内容选择器，根据实际结构调整
    content = '\n'.join([tag.text.strip() for tag in content_tags])
    return title, content

main_html = get_page_content(base_url)
if main_html:
    blog_links = parse_blog_links(main_html)
    print(f'Found {len(blog_links)} blog links.')
    count = 0
    for link in blog_links:
        article_html = get_page_content(link)
        if article_html:
            title, content = parse_blog_content(article_html)
            safe_title = title.replace('/', '_').replace(' ', '_')[:50]
            filepath = os.path.join(folder_name, safe_title + '.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(title + '\n\n' + content)
            count += 1
    print(f'Successfully saved {count} articles.')
else:
    print('Failed to fetch main page.')
