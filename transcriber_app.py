"""
Local YouTube transcriber with a web page.

Setup:  pip install yt-dlp faster-whisper
Run:    python transcriber_app.py
Then paste a YouTube link in the page that opens (http://localhost:8000).
"""
import os
import tempfile
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import yt_dlp
from faster_whisper import WhisperModel

MODEL_SIZE = "small"  # tiny / base / small / medium / large-v3
PORT = 8001  # new port so an old stuck copy can't answer

import av
print("Using av", av.__version__, "from", av.__file__)
print(f"Loading model '{MODEL_SIZE}' (first run downloads it)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
lock = threading.Lock()  # one transcription at a time

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YouTube Transcriber</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 16px; background: #111; color: #eee; }
  h1 { font-size: 1.4rem; }
  .row { display: flex; gap: 8px; margin: 12px 0; }
  input { flex: 1; padding: 10px; background: #222; color: #eee; border: 1px solid #444; border-radius: 6px; }
  button { padding: 10px 16px; background: #2563eb; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  #status { color: #9ca3af; margin: 8px 0; }
  #out { white-space: pre-wrap; background: #1a1a1a; padding: 12px; border-radius: 8px; min-height: 140px; }
</style>
</head>
<body>
<h1>YouTube Transcriber</h1>
<div class="row">
  <input id="url" placeholder="Paste YouTube link and press Enter">
  <button id="go">Transcribe</button>
  <button id="save" disabled>Save .txt</button>
</div>
<div id="status">Idle.</div>
<div id="out"></div>

<script>
const $ = id => document.getElementById(id);
let title = "transcript";

async function run() {
  const url = $("url").value.trim();
  if (!url) return;
  $("go").disabled = true;
  $("save").disabled = true;
  $("out").textContent = "";
  $("status").textContent = "Downloading audio...";
  title = "transcript";

  try {
    const res = await fetch("/transcribe?url=" + encodeURIComponent(url));
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\\n");
      buf = lines.pop();
      for (const line of lines) handle(line);
    }
    if (buf) handle(buf);
    if ($("status").textContent.startsWith("Error")) {}
    else $("status").textContent = "Done.";
    $("save").disabled = !$("out").textContent;
  } catch (e) {
    $("status").textContent = "Error: " + e.message;
  }
  $("go").disabled = false;
}

function handle(line) {
  if (line.startsWith("##TITLE ")) {
    title = line.slice(8);
    $("status").textContent = "Transcribing: " + title;
  } else if (line.startsWith("##ERROR ")) {
    $("status").textContent = "Error: " + line.slice(8);
  } else {
    $("out").textContent += line + "\\n";
  }
}

$("go").onclick = run;
$("url").addEventListener("keydown", e => { if (e.key === "Enter") run(); });
$("save").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([$("out").textContent], { type: "text/plain" }));
  a.download = title.replace(/[\\\\/:*?"<>|]+/g, "_").slice(0, 100) + ".txt";
  a.click();
};
</script>
</body>
</html>
"""


def fmt(t):
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_line(self, text):
        self.wfile.write((text + "\n").encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/transcribe":
            url = parse_qs(parsed.query).get("url", [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                with lock, tempfile.TemporaryDirectory() as tmp:
                    opts = {
                        "format": "bestaudio/best",
                        "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
                        "quiet": True,
                        "noplaylist": True,
                    }
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        path = ydl.prepare_filename(info)
                    self.send_line("##TITLE " + info.get("title", "transcript"))
                    segments, _ = model.transcribe(path, vad_filter=True)
                    for seg in segments:
                        self.send_line(f"[{fmt(seg.start)}] {seg.text.strip()}")
            except Exception as e:
                traceback.print_exc()
                try:
                    self.send_line("##ERROR " + str(e).replace("\n", " "))
                except Exception:
                    pass
            return

        self.send_error(404)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Ready at {url} (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
