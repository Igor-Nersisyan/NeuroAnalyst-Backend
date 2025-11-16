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
    logger.info(f"📄 Начинаю загрузку Google Doc: {gdoc_url}")
    try:
        r = requests.get(gdoc_url, timeout=30)
        logger.info(f"📄 Статус ответа: {r.status_code}")
        if r.status_code != 200:
            logger.error(f"❌ Google Doc вернул статус {r.status_code}")
            logger.error(f"❌ Ответ: {r.text[:500]}")
            raise ValueError(f"Ошибка загрузки Google Doc: {r.status_code}")
        text = r.text.strip()
        logger.info(f"📄 Загружен Google Doc ({len(text):,} символов)")
        if len(text) < 100:
            logger.warning(f"⚠️ Подозрительно короткий документ: {text[:100]}")
        return text
    except requests.RequestException as e:
        logger.error(f"❌ Ошибка при загрузке Google Doc: {e}")
        raise ValueError(f"Ошибка загрузки документа: {e}")
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}", exc_info=True)
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
    logger.info(f"🔎 Начинаю парсинг: {start_url}")
    logger.info(f"🔎 Параметры: max_pages={max_pages}, depth={depth}")
    
    visited, queue = set(), [(start_url, 0)]
    pages = []

    while queue and len(pages) < max_pages:
        url, d = queue.pop(0)
        if url in visited or d > depth:
            continue

        visited.add(url)
        logger.info(f"🌐 [{len(pages)+1}/{max_pages}]: {url}")

        try:
            logger.info(f"🌐 Делаю запрос к {url}...")
            r = requests.get(url, timeout=30, headers={"User-Agent": "NeuroAnalystBot/1.0"})
            logger.info(f"🌐 Статус: {r.status_code}")
            
            if r.status_code != 200:
                logger.warning(f"⚠️ Пропускаю {url}: статус {r.status_code}")
                continue

            logger.info(f"🌐 Парсинг HTML...")
            soup = BeautifulSoup(r.text, "html.parser")
            for s in soup(["script", "style", "noscript"]):
                s.extract()

            title = soup.title.string.strip() if soup.title else ""
            text = soup.get_text("\n", strip=True)[:20000]
            logger.info(f"🌐 Извлечено {len(text)} символов текста")

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

            logger.info(f"🌐 Найдено {len(links)} ссылок")
            
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

        except requests.Timeout:
            logger.error(f"⚠️ Таймаут при загрузке {url}")
        except requests.RequestException as e:
            logger.error(f"⚠️ Ошибка сети для {url}: {e}")
        except Exception as e:
            logger.error(f"⚠️ Ошибка парсинга {url}: {e}", exc_info=True)

    logger.info(f"✅ Парсинг завершен. Собрано {len(pages)} страниц")
    return {"start_url": start_url, "pages": pages, "count": len(pages)}


# -------------------------------
# 🤖 Модели
# -------------------------------
def call_main_model(client, prompt_text, site_data):
    logger.info("🤖 Запрос к gpt-5-mini...")
    logger.info(f"🤖 Размер промпта: {len(prompt_text)} символов")
    logger.info(f"🤖 Количество страниц в site_data: {site_data.get('count', 0)}")
    
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": json.dumps({"site": site_data}, ensure_ascii=False)},
    ]
    
    total_chars = len(prompt_text) + len(json.dumps(site_data))
    logger.info(f"🤖 Общий размер запроса: {total_chars:,} символов")
    
    try:
        logger.info("🤖 Отправляю запрос к OpenAI...")
        # БЕЗ ТАЙМАУТОВ - пусть ждет сколько надо
        resp = client.chat.completions.create(model="gpt-5-mini", messages=messages)
        logger.info(f"✅ Ответ получен от gpt-5-mini")
        logger.info(f"✅ Токены: {resp.usage.total_tokens:,} (prompt: {resp.usage.prompt_tokens:,}, completion: {resp.usage.completion_tokens:,})")
        return resp
    except Exception as e:
        logger.error(f"❌ Ошибка при вызове gpt-5-mini: {e}", exc_info=True)
        raise


def call_followup_model(client, followup_prompt_text, json_payload):
    logger.info("💬 Follow-up запрос...")
    logger.info(f"💬 Размер промпта: {len(followup_prompt_text)} символов")
    logger.info(f"💬 User instruction: {json_payload.get('user_instruction', 'НЕТ')[:100]}")
    
    messages = [
        {"role": "system", "content": followup_prompt_text},
        {"role": "user", "content": json.dumps(json_payload, ensure_ascii=False)},
    ]
    
    total_chars = len(followup_prompt_text) + len(json.dumps(json_payload))
    logger.info(f"💬 Общий размер запроса: {total_chars:,} символов")
    
    try:
        logger.info("💬 Отправляю follow-up запрос к OpenAI...")
        # БЕЗ ТАЙМАУТОВ - пусть ждет сколько надо
        resp = client.chat.completions.create(model="gpt-5-mini", messages=messages)
        logger.info(f"✅ Follow-up ответ получен от gpt-5-mini")
        logger.info(f"✅ Токены: {resp.usage.total_tokens:,} (prompt: {resp.usage.prompt_tokens:,}, completion: {resp.usage.completion_tokens:,})")
        return resp
    except Exception as e:
        logger.error(f"❌ Ошибка при вызове follow-up gpt-5-mini: {e}", exc_info=True)
        raise


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
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("❌ OPENAI_API_KEY не установлен в переменных окружения!")
    raise ValueError("OPENAI_API_KEY не найден в переменных окружения")

logger.info(f"🔑 OpenAI API key загружен (последние 4 символа: ...{api_key[-4:]})")
OPENAI_CLIENT = OpenAI(api_key=api_key)
logger.info("✅ OpenAI клиент инициализирован")


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
    
    logger.info(f"📝 site_url: {site_url}")
    logger.info(f"📝 existing_sid: {existing_sid}")

    if not site_url:
        logger.warning("⚠️ Отсутствует site_url")
        return jsonify({"error": "Нужно указать site_url"}), 400

    # Переиспользуем session_id если передан и существует
    if existing_sid and existing_sid in STORE:
        sid = existing_sid
        logger.info(f"♻️ Переиспользую session_id: {sid}")
    else:
        sid = str(uuid.uuid4())
        logger.info(f"🆕 Новый session_id: {sid}")

    try:
        logger.info("📄 Шаг 1: Загружаю промпт из Google Doc...")
        main_prompt = fetch_gdoc_text(MAIN_PROMPT_URL)
        logger.info(f"📄 Промпт загружен: {len(main_prompt)} символов")
        
        logger.info("🌐 Шаг 2: Начинаю парсинг сайта...")
        site_data = crawl_site(site_url)
        logger.info(f"🌐 Парсинг завершен: {site_data['count']} страниц")
        
        logger.info("🤖 Шаг 3: Отправляю данные в GPT...")
        resp = call_main_model(OPENAI_CLIENT, main_prompt, site_data)
        
        logger.info("🤖 Шаг 4: Извлекаю ответ...")
        model_output = resp.choices[0].message.content
        logger.info(f"🤖 Размер ответа: {len(model_output)} символов")

    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА в /analyze: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    # ПОЛНАЯ перезапись сессии
    logger.info("💾 Сохраняю сессию...")
    STORE[sid] = {
        "site": site_data,
        "first_output": model_output,
        "last_followup": None,
        "history": [],
        "created_at": datetime.now()
    }
    logger.info(f"💾 Сессия {sid} сохранена")

    logger.info(f"✅ Анализ завершён успешно")
    logger.info("=" * 60)

    return jsonify({
        "session_id": sid,
        "result": model_output,
        "pages": site_data["count"]
    })


@app.route("/followup", methods=["POST"])
def followup():
    logger.info("=" * 60)
    logger.info("💬 /followup")
    
    data = request.json or {}
    sid = data.get("session_id")
    user_instruction = data.get("followup_prompt")
    
    logger.info(f"📝 session_id: {sid}")
    logger.info(f"📝 user_instruction: {user_instruction[:100] if user_instruction else 'НЕТ'}")

    if not sid or sid not in STORE:
        logger.warning(f"⚠️ session_id {sid} не найден в STORE")
        logger.info(f"⚠️ Доступные сессии: {list(STORE.keys())}")
        return jsonify({"error": "session_id не найден"}), 404

    sess = STORE[sid]
    logger.info(f"📂 Сессия найдена. История: {len(sess.get('history', []))} сообщений")

    try:
        logger.info("📄 Загружаю follow-up промпт...")
        followup_prompt_text = fetch_gdoc_text(FOLLOWUP_PROMPT_URL)
        logger.info(f"📄 Follow-up промпт загружен: {len(followup_prompt_text)} символов")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки follow-up промпта: {e}", exc_info=True)
        return jsonify({"error": f"Ошибка загрузки промпта: {e}"}), 500

    payload = {
        "first_output": sess.get("first_output"),
        "last_followup": sess.get("last_followup"),
        "conversation_history": sess.get("history", []),
        "user_instruction": user_instruction
    }
    
    logger.info(f"📦 Размер payload: {len(json.dumps(payload))} символов")

    try:
        logger.info("🤖 Отправляю follow-up запрос в GPT...")
        resp = call_followup_model(OPENAI_CLIENT, followup_prompt_text, payload)
        
        logger.info("🤖 Извлекаю ответ...")
        model_text = resp.choices[0].message.content
        logger.info(f"🤖 Размер ответа: {len(model_text)} символов")

        logger.info("💾 Обновляю сессию...")
        sess["last_followup"] = model_text
        sess["history"].append({"role": "user", "content": user_instruction})
        sess["history"].append({"role": "assistant", "content": model_text})
        logger.info(f"💾 История обновлена: {len(sess['history'])} сообщений")

    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА в /followup: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    logger.info(f"✅ Follow-up завершён успешно")
    logger.info("=" * 60)

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
