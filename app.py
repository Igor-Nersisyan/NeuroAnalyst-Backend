# ==========================================================
#  NeuroAnalyst Backend — Production
# ==========================================================

import os, re, json, time, uuid, logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import tldextract
import requests
from openai import OpenAI


# -------------------------------
# ⚙️ Конфигурация
# -------------------------------
# ФИКС: Используем export?format=txt для программного доступа к Google Docs
MAIN_PROMPT_URL = "https://docs.google.com/document/d/1DtA6CzcNeoZSDwj043YmE84XMnv1LAp_Z3MWxP8n55M/export?format=txt"
FOLLOWUP_PROMPT_URL = "https://docs.google.com/document/d/12nwxCLf4Gk4daR7ecRA04rZe-RToNb8-TAtERzY4o0E/export?format=txt"

SESSION_TTL_HOURS = 24
MAX_SESSIONS = 100

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("neuro-analyst")


# -------------------------------
# 📄 Загрузка Google Docs
# -------------------------------
def fetch_gdoc_text(gdoc_url: str) -> str:
    try:
        r = requests.get(gdoc_url)
        if r.status_code != 200:
            raise ValueError(f"Ошибка загрузки Google Doc: {r.status_code}")
        text = r.text.strip()
        logger.info(f"📄 Загружен Google Doc ({len(text):,} символов)")
        return text
    except Exception as e:
        raise ValueError(f"Ошибка загрузки документа: {e}")


# -------------------------------
# 🌐 Нормализация ссылок
# -------------------------------
def normalize_link(base, href: str):
    if not href or not isinstance(href, str):
        return None

    href = href.strip()

    bad_prefixes = (
        "mailto:", "tel:", "javascript:", "whatsapp:", "viber:",
        "tg:", "#", "sms:", "skype:",
    )
    if href.startswith(bad_prefixes):
        return None

    if href.startswith("http://") or href.startswith("https://"):
        return href.split("#")[0]

    if href.startswith("//"):
        return "https:" + href.split("#")[0]

    return urljoin(base, href.split("#")[0])


# -------------------------------
# 🔎 Парсинг сайта
# -------------------------------
def same_domain(a, b):
    try:
        return tldextract.extract(a).registered_domain == tldextract.extract(b).registered_domain
    except:
        return False


def safe_json(obj):
    try:
        json.dumps(obj)
        return obj
    except:
        return str(obj)


def crawl_site(start_url, max_pages=25, depth=1):
    logger.info(f"🔎 Парсинг: {start_url}")
    
    visited, queue = set(), [(start_url, 0)]
    pages = []

    while queue and len(pages) < max_pages:
        url, d = queue.pop(0)
        if url in visited or d > depth:
            continue

        visited.add(url)
        logger.info(f"🌐 [{len(pages)+1}/{max_pages}]: {url}")

        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "NeuroAnalystBot/1.0"})
            if r.status_code != 200:
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            for s in soup(["script", "style", "noscript"]):
                s.extract()

            title = soup.title.string.strip() if soup.title else ""
            text = soup.get_text("\n", strip=True)[:20000]

            meta = {
                m.get("name", m.get("property", "")): m.get("content", "")
                for m in soup.find_all("meta")
                if m.get("name") or m.get("property")
            }

            links = []
            for a in soup.find_all("a", href=True):
                link = normalize_link(url, a["href"])
                if link and same_domain(start_url, link):
                    links.append(link)

            pages.append({
                "url": url,
                "title": title,
                "meta": safe_json(meta),
                "text": text,
                "links": links
            })

            for l in links:
                if l not in visited:
                    queue.append((l, d + 1))

        except Exception as e:
            logger.error(f"⚠️ Ошибка {url}: {e}")

    logger.info(f"✅ Собрано {len(pages)} страниц")
    return {"start_url": start_url, "pages": pages, "count": len(pages)}


# -------------------------------
# 🤖 Модели
# -------------------------------
def call_main_model(client, prompt_text, site_data):
    logger.info("🤖 Запрос к gpt-5-mini...")
    
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": json.dumps({"site": site_data}, ensure_ascii=False)},
    ]
    
    resp = client.chat.completions.create(model="gpt-5-mini", messages=messages)
    logger.info(f"✅ Ответ получен ({resp.usage.total_tokens:,} токенов)")
    return resp


def call_followup_model(client, followup_prompt_text, json_payload):
    logger.info("💬 Follow-up запрос...")
    
    messages = [
        {"role": "system", "content": followup_prompt_text},
        {"role": "user", "content": json.dumps(json_payload, ensure_ascii=False)},
    ]
    
    resp = client.chat.completions.create(model="gpt-5-mini", messages=messages)
    logger.info(f"✅ Ответ получен ({resp.usage.total_tokens:,} токенов)")
    return resp


# -------------------------------
# 🗄️ Управление сессиями
# -------------------------------
STORE = {}

def cleanup_old_sessions():
    now = datetime.now()
    to_delete = [
        sid for sid, sess in STORE.items()
        if sess.get("created_at") and (now - sess["created_at"]) > timedelta(hours=SESSION_TTL_HOURS)
    ]
    
    for sid in to_delete:
        del STORE[sid]
    
    if to_delete:
        logger.info(f"🧹 Очищено {len(to_delete)} старых сессий")


def limit_sessions():
    if len(STORE) > MAX_SESSIONS:
        sorted_sessions = sorted(STORE.items(), key=lambda x: x[1].get("created_at", datetime.min))
        to_delete = len(STORE) - MAX_SESSIONS
        
        for sid, _ in sorted_sessions[:to_delete]:
            del STORE[sid]
        
        logger.info(f"🧹 Очищено {to_delete} сессий (лимит)")


# -------------------------------
# 🌍 Flask API
# -------------------------------
app = Flask(__name__)
CORS(app)

# Получаем OpenAI client из env
OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "alive",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sessions": len(STORE)
    }), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    logger.info("=" * 60)
    logger.info("🆕 /analyze")
    
    # Детальное логирование запроса
    logger.info(f"Request data: {request.json}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    cleanup_old_sessions()
    limit_sessions()
    
    data = request.json or {}
    site_url = data.get("site_url")
    existing_sid = data.get("session_id")

    if not site_url:
        return jsonify({"error": "Нужно указать site_url"}), 400

    # Переиспользуем session_id если передан и существует
    if existing_sid and existing_sid in STORE:
        sid = existing_sid
        logger.info(f"♻️ Переиспользую {sid}")
    else:
        sid = str(uuid.uuid4())
        logger.info(f"🆕 Новый {sid}")

    try:
        main_prompt = fetch_gdoc_text(MAIN_PROMPT_URL)
        site_data = crawl_site(site_url)
        resp = call_main_model(OPENAI_CLIENT, main_prompt, site_data)
        model_output = resp.choices[0].message.content

    except Exception as e:
        logger.error(f"❌ ОШИБКА: {e}", exc_info=True)  # exc_info=True для полного traceback
        return jsonify({"error": str(e)}), 500

    # ПОЛНАЯ перезапись сессии
    STORE[sid] = {
        "site": site_data,
        "first_output": model_output,
        "last_followup": None,
        "history": [],
        "created_at": datetime.now()
    }

    logger.info(f"✅ Анализ завершён")
    logger.info("=" * 60)

    return jsonify({
        "session_id": sid,
        "result": model_output,
        "pages": site_data["count"]
    })


@app.route("/followup", methods=["POST"])
def followup():
    logger.info("💬 /followup")
    
    data = request.json or {}
    sid = data.get("session_id")
    user_instruction = data.get("followup_prompt")

    if not sid or sid not in STORE:
        return jsonify({"error": "session_id не найден"}), 404

    sess = STORE[sid]

    try:
        followup_prompt_text = fetch_gdoc_text(FOLLOWUP_PROMPT_URL)
    except Exception as e:
        return jsonify({"error": f"Ошибка загрузки промпта: {e}"}), 500

    payload = {
        "first_output": sess.get("first_output"),
        "last_followup": sess.get("last_followup"),
        "conversation_history": sess.get("history", []),
        "user_instruction": user_instruction
    }

    try:
        resp = call_followup_model(OPENAI_CLIENT, followup_prompt_text, payload)
        model_text = resp.choices[0].message.content

        sess["last_followup"] = model_text
        sess["history"].append({"role": "user", "content": user_instruction})
        sess["history"].append({"role": "assistant", "content": model_text})

    except Exception as e:
        logger.error(f"❌ ОШИБКА: {e}", exc_info=True)  # exc_info=True для полного traceback
        return jsonify({"error": str(e)}), 500

    logger.info(f"✅ Follow-up завершён")

    return jsonify({"result": model_text})


@app.route("/clear-chat", methods=["POST"])
def clear_chat():
    data = request.json or {}
    sid = data.get("session_id")

    if not sid or sid not in STORE:
        return jsonify({"error": "session_id не найден"}), 404

    sess = STORE[sid]
    messages_count = len(sess.get("history", []))
    
    sess["history"] = []
    sess["last_followup"] = None
    
    logger.info(f"🧹 Очищен чат ({messages_count} сообщений)")

    return jsonify({
        "status": "success",
        "message": f"История чата очищена ({messages_count} сообщений)",
        "session_id": sid
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
