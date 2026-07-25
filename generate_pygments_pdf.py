import os
import sys
import subprocess
import html
import datetime
import shutil
import pygments
from pygments.lexers import get_lexer_for_filename, TextLexer
from pygments.formatters import HtmlFormatter
from pygments import highlight

def generate_pdf():
    repo_root = os.path.abspath(r'c:\Users\USER-30\Indoor-GPS-')
    output_html = os.path.join(repo_root, 'repo_pygments.html')
    output_pdf = os.path.join(repo_root, 'Indoor-GPS_Repository_Codebase.pdf')
    artifact_dir = r'C:\Users\USER-30\.gemini\antigravity-ide\brain\89cb2f9e-faa8-45b4-bc61-faec937fca05'
    artifact_pdf = os.path.join(artifact_dir, 'Indoor-GPS_Repository_Codebase.pdf')

    print('Gathering git files...')
    cmd = ['git', 'ls-files']
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if res.returncode != 0:
        print('git ls-files failed:', res.stderr)
        return

    tracked = res.stdout.strip().split('\n')
    binary_exts = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.dylib', '.pyc'}

    files_data = []
    total_lines = 0
    total_bytes = 0

    style_name = 'github-dark'
    formatter = HtmlFormatter(style=style_name, linenos='inline', cssclass='highlight', wrapcode=True)
    pygments_css = formatter.get_style_defs('.highlight')

    for idx, rel_path in enumerate(tracked):
        rel_path = rel_path.strip()
        if not rel_path:
            continue
        ext = os.path.splitext(rel_path)[1].lower()
        if ext in binary_exts:
            continue
        full_path = os.path.join(repo_root, rel_path)
        if not os.path.isfile(full_path):
            continue

        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                code = f.read()
            lines = code.count('\n') + (1 if code else 0)
            size = len(code.encode('utf-8'))
            total_lines += lines
            total_bytes += size

            try:
                lexer = get_lexer_for_filename(rel_path, code)
            except Exception:
                lexer = TextLexer()

            highlighted_code = highlight(code, lexer, formatter)
            anchor_id = f'file_{idx}'
            files_data.append({
                'id': anchor_id,
                'path': rel_path,
                'lines': lines,
                'size': size,
                'lexer': lexer.name,
                'code_html': highlighted_code
            })
        except Exception as e:
            print(f'Error processing {rel_path}: {e}')

    print(f'Processed {len(files_data)} files, {total_lines:,} lines, {total_bytes / (1024*1024):.2f} MB')

    html_parts = []
    html_parts.append(f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Indoor-GPS Repository Codebase</title>
<style>
@page {{
    size: A4 portrait;
    margin: 10mm;
}}
body {{
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
    font-size: 12px;
    line-height: 1.4;
}}
a {{
    color: #58a6ff;
    text-decoration: none;
}}
a:hover {{
    text-decoration: underline;
}}
.cover-page {{
    padding: 60px 20px;
    text-align: center;
    page-break-after: always;
    break-after: page;
}}
.cover-title {{
    font-size: 32px;
    font-weight: 800;
    color: #58a6ff;
    margin-bottom: 12px;
}}
.cover-subtitle {{
    font-size: 18px;
    color: #8b949e;
    margin-bottom: 40px;
}}
.stats-box {{
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 24px 36px;
    text-align: left;
    margin-bottom: 30px;
}}
.stats-item {{
    font-size: 14px;
    margin: 8px 0;
    color: #c9d1d9;
}}
.toc-section {{
    page-break-after: always;
    break-after: page;
    padding: 20px;
}}
.toc-title {{
    font-size: 22px;
    font-weight: 700;
    color: #f0f6fc;
    border-bottom: 1px solid #30363d;
    padding-bottom: 8px;
    margin-bottom: 20px;
}}
.toc-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
}}
.toc-table th, .toc-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid #21262d;
    text-align: left;
}}
.toc-table th {{
    background: #161b22;
    color: #8b949e;
}}
.file-container {{
    page-break-after: always;
    break-after: page;
    padding: 10px 0;
}}
.file-header {{
    background: #161b22;
    border: 1px solid #30363d;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 10px 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.file-path {{
    font-family: "Cascadia Code", Consolas, Monaco, monospace;
    font-weight: bold;
    font-size: 13px;
    color: #f0f6fc;
}}
.file-meta {{
    font-size: 11px;
    color: #8b949e;
}}
.highlight-wrap {{
    border: 1px solid #30363d;
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
    background: #0d1117;
    overflow-x: auto;
}}
.highlight {{
    margin: 0;
    padding: 10px;
    font-family: "Cascadia Code", Consolas, "Fira Code", Monaco, monospace;
    font-size: 10px;
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-all;
}}
{pygments_css}
.lineno {{
    color: #484f58;
    padding-right: 10px;
    user-select: none;
}}
</style>
</head>
<body>

<div class="cover-page">
    <div class="cover-title">Indoor-GPS Repository Codebase</div>
    <div class="cover-subtitle">Visible Light Communication & Localization (VLCL) AI Digital Twin</div>
    <div class="stats-box">
        <div class="stats-item"><strong>Generated Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        <div class="stats-item"><strong>Total Tracked Files:</strong> {len(files_data)}</div>
        <div class="stats-item"><strong>Total Code Lines:</strong> {total_lines:,}</div>
        <div class="stats-item"><strong>Total Code Size:</strong> {total_bytes / (1024*1024):.2f} MB</div>
        <div class="stats-item"><strong>Syntax Highlighter:</strong> Pygments v{pygments.__version__} ({style_name} theme)</div>
    </div>
</div>

<div class="toc-section">
    <div class="toc-title">Table of Contents ({len(files_data)} Files)</div>
    <table class="toc-table">
        <thead>
            <tr>
                <th>#</th>
                <th>File Path</th>
                <th>Language / Lexer</th>
                <th>Lines</th>
                <th>Size</th>
            </tr>
        </thead>
        <tbody>
''')

    for idx, f in enumerate(files_data, 1):
        html_parts.append(f'''
            <tr>
                <td>{idx}</td>
                <td><a href="#{f['id']}">{html.escape(f['path'])}</a></td>
                <td>{html.escape(f['lexer'])}</td>
                <td>{f['lines']:,}</td>
                <td>{f['size'] / 1024:.1f} KB</td>
            </tr>
        ''')

    html_parts.append('''
        </tbody>
    </table>
</div>
''')

    for f in files_data:
        html_parts.append(f'''
<div class="file-container" id="{f['id']}">
    <div class="file-header">
        <span class="file-path">{html.escape(f['path'])}</span>
        <span class="file-meta">{html.escape(f['lexer'])} | {f['lines']:,} lines | {f['size'] / 1024:.1f} KB</span>
    </div>
    <div class="highlight-wrap">
        {f['code_html']}
    </div>
</div>
''')

    html_parts.append('</body></html>')

    full_html = ''.join(html_parts)
    print(f'Writing HTML ({len(full_html) / (1024*1024):.2f} MB)...')
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print('Converting to PDF using Chrome Headless...')
    chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
    cmd = [chrome, '--headless=new', f'--print-to-pdf={output_pdf}', '--no-pdf-header-footer', output_html]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode == 0 and os.path.exists(output_pdf):
        pdf_sz = os.path.getsize(output_pdf)
        print(f'PDF successfully created: {output_pdf} ({pdf_sz / (1024*1024):.2f} MB)')
        if os.path.exists(artifact_dir):
            shutil.copy2(output_pdf, artifact_pdf)
            print(f'Copied PDF to artifacts: {artifact_pdf}')
    else:
        print('PDF creation failed!', res.returncode, res.stderr)

if __name__ == '__main__':
    generate_pdf()
