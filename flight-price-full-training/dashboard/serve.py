"""Local server for Flight Price Dashboard with Ollama reverse proxy.
Zero dependencies (Python standard library only).
Resolves browser CORS restrictions when calling Ollama from localhost.
"""
import os
import sys
import webbrowser
import urllib.request
import urllib.error
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = 8000
OLLAMA_TARGET = "http://localhost:11434"
DIR = os.path.dirname(os.path.abspath(__file__))


class DashboardProxyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy_to_ollama()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self._proxy_to_ollama()
        else:
            self.send_error(404)

    def _proxy_to_ollama(self):
        url = f"{OLLAMA_TARGET}{self.path}"
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else None

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "User-Agent": "DashboardProxy/1.0"
            },
            method=self.command
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as res:
                self.send_response(res.status)
                self.send_header("Access-Control-Allow-Origin", "*")
                for k, v in res.getheaders():
                    if k.lower() not in ["access-control-allow-origin", "transfer-encoding"]:
                        self.send_header(k, v)
                self.end_headers()

                # Stream response back chunk by chunk
                while True:
                    chunk = res.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            err_json = f'{{"error": "Không thể kết nối Ollama: {str(e)}"}}'.encode("utf-8")
            self.wfile.write(err_json)


def run():
    os.chdir(DIR)
    port = PORT
    server = None
    for p in range(PORT, PORT + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), DashboardProxyHandler)
            port = p
            break
        except OSError:
            continue

    if not server:
        print(f"Lỗi: Không tìm thấy cổng trống từ {PORT} đến {PORT+10}")
        sys.exit(1)

    url = f"http://localhost:{port}/index.html"
    print("=" * 65)
    print(f"  ✈️  FLIGHT PRICE DASHBOARD + AI COPILOT (QWEN 2.5)")
    print("=" * 65)
    print(f"  - Web Dashboard: {url}")
    print(f"  - Ollama Target: {OLLAMA_TARGET}")
    print(f"  - Đang tự động mở trình duyệt...")
    print(f"  - Nhấn Ctrl + C để dừng server.")
    print("=" * 65)

    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
        server.server_close()


if __name__ == "__main__":
    run()
