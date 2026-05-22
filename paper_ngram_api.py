from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import requests
import re
import glob
from collections import Counter, defaultdict
import sqlite3
import uuid
from datetime import datetime
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import csv
import io
from fastapi.responses import Response


class PaperNgramRuntime:
    def __init__(self, model_root: str):
        self.model_root = model_root
        self.split_folders = self._find_best_split_folders()
        self.models = self._load_all_models()

        self.stopwords = {
            "i", "me", "my", "mine", "you", "your", "yours",
            "he", "him", "his", "she", "her", "hers",
            "it", "its", "we", "us", "our", "ours",
            "they", "them", "their", "theirs",
            "a", "an", "the",
            "and", "or", "but",
            "to", "of", "in", "on", "at", "for", "from", "by", "with",
            "is", "am", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "have", "has", "had",
            "this", "that", "these", "those",
            "as", "if", "then", "than"
        }

    def _find_best_split_folders(self):
        folders = sorted([
            d for d in glob.glob(os.path.join(self.model_root, "multiclass-*-0"))
            if os.path.isdir(d)
        ])
        return folders

    def _parse_line(self, raw_line: str):
        raw_line = raw_line.strip()
        if not raw_line:
            return None

        if "\t" in raw_line:
            phrase, weight = raw_line.rsplit("\t", 1)
        else:
            phrase, weight = raw_line, "1.0"

        phrase = phrase.lstrip(",").strip().lower()
        phrase = re.sub(r"\s+", " ", phrase)

        if not phrase:
            return None

        try:
            weight = float(weight)
        except:
            weight = 1.0

        return phrase, weight

    def _load_split_folder(self, folder_path: str):
        split_model = {}

        txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
        for fp in txt_files:
            label = os.path.splitext(os.path.basename(fp))[0]
            lexicon = {}

            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    parsed = self._parse_line(line)
                    if parsed is None:
                        continue

                    phrase, weight = parsed
                    if phrase not in lexicon or weight > lexicon[phrase]:
                        lexicon[phrase] = weight

            split_model[label] = lexicon

        return split_model

    def _load_all_models(self):
        all_models = []
        for folder in self.split_folders:
            split_model = self._load_split_folder(folder)
            all_models.append({
                "folder": folder,
                "labels": split_model
            })
        return all_models

    def _extract_token_spans(self, text: str):
        pattern = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?%?")
        spans = []

        for m in pattern.finditer(text):
            spans.append({
                "token": m.group(0).lower(),
                "start": m.start(),
                "end": m.end()
            })

        return spans

    def _extract_text_ngrams(self, text: str):
        token_spans = self._extract_token_spans(text)
        candidates = []

        for item in token_spans:
            tok = item["token"]
            if len(tok) < 3:
                continue
            if tok in self.stopwords:
                continue

            candidates.append({
                "phrase": tok,
                "start": item["start"],
                "end": item["end"],
                "tokens": 1
            })

        for i in range(len(token_spans) - 1):
            left = token_spans[i]
            right = token_spans[i + 1]

            phrase = f"{left['token']} {right['token']}"
            candidates.append({
                "phrase": phrase,
                "start": left["start"],
                "end": right["end"],
                "tokens": 2
            })

        seen = set()
        unique_candidates = []
        for c in candidates:
            key = (c["phrase"], c["start"], c["end"])
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(c)

        return unique_candidates

    def _score_one_split(self, split_model, text: str):
        label_scores = Counter()
        label_matches = defaultdict(list)

        candidates = self._extract_text_ngrams(text)

        for label, lexicon in split_model["labels"].items():
            for cand in candidates:
                phrase = cand["phrase"]
                if phrase not in lexicon:
                    continue

                weight = lexicon[phrase]
                token_bonus = 2.0 if cand["tokens"] == 2 else 1.0
                score = weight * token_bonus

                label_scores[label] += score
                label_matches[label].append({
                    "phrase": text[cand["start"]:cand["end"]],
                    "start": cand["start"],
                    "end": cand["end"],
                    "label": label,
                    "score": score,
                    "tokens": cand["tokens"]
                })

        return label_scores, label_matches

    def _deduplicate_matches(self, matches):
        if not matches:
            return []

        matches = sorted(
            matches,
            key=lambda m: (-m["tokens"], -m["score"], m["start"], -(m["end"] - m["start"]))
        )

        chosen = []
        occupied = []

        for m in matches:
            overlap = False
            for s, e in occupied:
                if not (m["end"] <= s or m["start"] >= e):
                    overlap = True
                    break

            if not overlap:
                chosen.append(m)
                occupied.append((m["start"], m["end"]))

        chosen = sorted(chosen, key=lambda x: x["start"])
        return chosen

    def _ensemble_scores(self, text: str):
        ensemble_scores = Counter()
        ensemble_matches = defaultdict(list)

        for split_model in self.models:
            split_scores, split_matches = self._score_one_split(split_model, text)

            for label, score in split_scores.items():
                ensemble_scores[label] += score

            for label, matches in split_matches.items():
                ensemble_matches[label].extend(matches)

        return ensemble_scores, ensemble_matches

    def analyze(self, text: str):
        """
        Analyze one NPC sentence.

        保留原来的字段：
        - label
        - score
        - keywords

        新增研究用字段：
        - top_labels
        - top1_label / top1_score
        - top2_label / top2_score
        - score_margin
        - raw_keyword_count
        - keyword_count
        - text_word_count
        - keyword_density
        - assessment_valid
        - invalid_reason
        """

        # ---------- 小工具：计算词数 ----------
        def count_words(t: str) -> int:
            return len(self._extract_token_spans(t or ""))

        # ---------- 小工具：统一返回无效结果 ----------
        def invalid_response(reason: str, top_labels=None):
            word_count = count_words(text)
            return {
                "label": "No_Distortion",
                "score": 0.0,

                "top_labels": top_labels or [],

                "top1_label": "No_Distortion",
                "top1_score": 0.0,
                "top2_label": "",
                "top2_score": 0.0,
                "score_margin": 0.0,

                "raw_keyword_count": 0,
                "keyword_count": 0,
                "keywords": [],

                "text_word_count": word_count,
                "keyword_density": 0.0,

                "assessment_valid": False,
                "invalid_reason": reason
            }

        # ---------- 1. 计算每个 distortion 的分数 ----------
        ensemble_scores, ensemble_matches = self._ensemble_scores(text)

        if not ensemble_scores:
            return invalid_response("no_model_score")

        # 不让 No_Distortion 参与 top1/top2 排名
        distortion_scores = [
            (label, score)
            for label, score in ensemble_scores.items()
            if label != "No_Distortion"
        ]

        distortion_scores = sorted(
            distortion_scores,
            key=lambda x: x[1],
            reverse=True
        )

        if not distortion_scores:
            return invalid_response("no_distortion_score")

        # ---------- 2. top1 / top2 / margin ----------
        best_label, best_score = distortion_scores[0]

        if len(distortion_scores) >= 2:
            second_label, second_score = distortion_scores[1]
        else:
            second_label, second_score = "", 0.0

        score_margin = float(best_score - second_score)

        top_labels = []
        for label, score in distortion_scores[:3]:
            top_labels.append({
                "label": label,
                "score": float(score)
            })

        # ---------- 3. 从 top3 distortion 中取候选关键词 ----------
        candidate_matches = []
        for label, score in distortion_scores[:3]:
            for m in ensemble_matches[label]:
                candidate_matches.append(m)

        # 去重：优先保留 bigram、高分、不重叠的 span
        keywords_raw = self._deduplicate_matches(candidate_matches)
        raw_keyword_count = len(keywords_raw)

        # ---------- 4. 过滤低价值关键词 ----------
        bad_phrases = {
            "to me",
            "me",
            "to",
            "will",
            "ever",
            "good",
            "happen"
        }

        filtered_keywords = []
        for k in keywords_raw:
            p = k["phrase"].strip().lower()
            if p in bad_phrases:
                continue
            filtered_keywords.append(k)

        # 最多返回 8 个 NLP 原始高亮候选
        filtered_keywords = filtered_keywords[:8]

        word_count = count_words(text)
        keyword_count = len(filtered_keywords)

        if word_count > 0:
            keyword_density = keyword_count / word_count
        else:
            keyword_density = 0.0

        # ---------- 5. 基础 NLP 可用性判断 ----------
        # 注意：这里还没有判断“是否等于当前游戏目标类型”
        # 目标类型匹配要在 Unity 端判断，因为 API 不知道当前关卡目标是什么。
        if best_score < 2.0:
            return {
                "label": "No_Distortion",
                "score": float(best_score),

                "top_labels": top_labels,

                "top1_label": best_label,
                "top1_score": float(best_score),
                "top2_label": second_label,
                "top2_score": float(second_score),
                "score_margin": score_margin,

                "raw_keyword_count": raw_keyword_count,
                "keyword_count": 0,
                "keywords": [],

                "text_word_count": word_count,
                "keyword_density": 0.0,

                "assessment_valid": False,
                "invalid_reason": "score_too_low"
            }

        if keyword_count == 0:
            return {
                "label": "No_Distortion",
                "score": float(best_score),

                "top_labels": top_labels,

                "top1_label": best_label,
                "top1_score": float(best_score),
                "top2_label": second_label,
                "top2_score": float(second_score),
                "score_margin": score_margin,

                "raw_keyword_count": raw_keyword_count,
                "keyword_count": 0,
                "keywords": [],

                "text_word_count": word_count,
                "keyword_density": 0.0,

                "assessment_valid": False,
                "invalid_reason": "no_keywords_after_filter"
            }

        # ---------- 6. 组织关键词输出 ----------
        keywords = [
            {
                "phrase": k["phrase"],
                "start": k["start"],
                "end": k["end"],
                "label": k["label"],
                "score": float(k.get("score", 0.0)),
                "tokens": int(k.get("tokens", 1))
            }
            for k in filtered_keywords
        ]

        # ---------- 7. 正常返回 ----------
        return {
            # 原有字段：Unity 老代码依然能读
            "label": best_label,
            "score": float(best_score),
            "keywords": keywords,

            # 新增字段：后面 Unity / BN 会用
            "top_labels": top_labels,

            "top1_label": best_label,
            "top1_score": float(best_score),
            "top2_label": second_label,
            "top2_score": float(second_score),
            "score_margin": score_margin,

            "raw_keyword_count": raw_keyword_count,
            "keyword_count": keyword_count,

            "text_word_count": word_count,
            "keyword_density": keyword_density,

            # 这里只代表 NLP 基础可用
            # 还不代表这个句子一定适合当前关卡
            "assessment_valid": True,
            "invalid_reason": ""
        }


app = FastAPI()

# ===== CORS: 以后 WebGL / itch.io 调用后端时需要 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 预测试阶段先全部允许；正式上线后可以改成你的网页域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== SQLite Research Database =====

DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(__file__))
DB_PATH = os.path.join(DATA_DIR, "research_data.sqlite")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def now_iso():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def parse_payload(payload_json: str):
    if not payload_json:
        return {}

    try:
        return json.loads(payload_json)
    except Exception:
        return {}
    
def make_csv_response(filename: str, fieldnames, rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for row in rows:
        writer.writerow(row)

    csv_text = output.getvalue()

    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


def init_research_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # 一次游玩 / 一个参与者 session
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        participant_id TEXT,
        condition TEXT,
        platform TEXT,
        app_version TEXT,
        consent INTEGER DEFAULT 0,
        started_at_server TEXT,
        ended_at_server TEXT,
        completed INTEGER DEFAULT 0,
        last_checkpoint TEXT
    )
    """)

    # 通用事件表：先用一个表最快跑通
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        participant_id TEXT,
        condition TEXT,
        event_type TEXT NOT NULL,

        client_time REAL,
        server_time TEXT NOT NULL,

        scene TEXT,
        map_id TEXT,
        zone_id TEXT,
        case_id TEXT,
        sentence_id TEXT,

        source_mode TEXT,
        target_distortion TEXT,

        payload_json TEXT
    )
    """)

    # 常用索引，后面导出数据会快一些
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_sentence ON events(sentence_id)")

    conn.commit()
    conn.close()


init_research_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.environ.get("MODEL_ROOT", BASE_DIR)

print(f"[MODEL] BASE_DIR={BASE_DIR}", flush=True)
print(f"[MODEL] MODEL_ROOT={MODEL_ROOT}", flush=True)
print(f"[MODEL] multiclass folders={glob.glob(os.path.join(MODEL_ROOT, 'multiclass*'))[:20]}", flush=True)

runtime = PaperNgramRuntime(MODEL_ROOT)


class AnalyzeRequest(BaseModel):
    text: str
        
class StartSessionRequest(BaseModel):
    participant_id: Optional[str] = ""
    condition: Optional[str] = "unknown"
    platform: Optional[str] = ""
    app_version: Optional[str] = ""
    consent: bool = False


class StartSessionResponse(BaseModel):
    session_id: str


class LogEventRequest(BaseModel):
    session_id: Optional[str] = ""
    participant_id: Optional[str] = ""
    condition: Optional[str] = ""

    event_type: str

    client_time: Optional[float] = 0.0

    scene: Optional[str] = ""
    map_id: Optional[str] = ""
    zone_id: Optional[str] = ""
    case_id: Optional[str] = ""
    sentence_id: Optional[str] = ""

    source_mode: Optional[str] = ""
    target_distortion: Optional[str] = ""

    # Unity 端传过来的完整 JSON 字符串
    payload_json: Optional[str] = "{}"


class CompleteSessionRequest(BaseModel):
    session_id: str
    last_checkpoint: Optional[str] = ""

class ChatMessageIn(BaseModel):
    role: str
    content: str


class GenerateDialogueRequest(BaseModel):
    messages: list[ChatMessageIn]
    temperature: float = 0.7
    max_tokens: int = 120


class GenerateDialogueResponse(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "ok", "message": "paper ngram api is running"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return runtime.analyze(req.text)

@app.post("/generate_dialogue", response_model=GenerateDialogueResponse)
def generate_dialogue(req: GenerateDialogueRequest):
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set on the server.")

    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()

    messages = []
    for m in req.messages:
        role = (m.role or "").strip()
        content = (m.content or "").strip()

        if not role or not content:
            continue

        messages.append({
            "role": role,
            "content": content
        })

    if not messages:
        raise RuntimeError("No valid messages were provided.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stream": False,

        # 游戏 NPC 对话不需要长推理，关闭 thinking 可以降低延迟和成本
        "thinking": {
            "type": "disabled"
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
    except Exception as e:
        raise RuntimeError(f"DeepSeek request failed: {e}")

    if r.status_code >= 400:
        raise RuntimeError(f"DeepSeek API error {r.status_code}: {r.text}")

    data = r.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected DeepSeek response: {data}")

    return GenerateDialogueResponse(text=text.strip())

@app.post("/start_session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    session_id = str(uuid.uuid4())

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO sessions (
        session_id,
        participant_id,
        condition,
        platform,
        app_version,
        consent,
        started_at_server,
        completed,
        last_checkpoint
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
    """, (
        session_id,
        req.participant_id or "",
        req.condition or "unknown",
        req.platform or "",
        req.app_version or "",
        1 if req.consent else 0,
        now_iso(),
        "session_start"
    ))

    cur.execute("""
    INSERT INTO events (
        session_id,
        participant_id,
        condition,
        event_type,
        client_time,
        server_time,
        scene,
        payload_json
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        req.participant_id or "",
        req.condition or "unknown",
        "session_start",
        0.0,
        now_iso(),
        "",
        json.dumps({
            "platform": req.platform,
            "app_version": req.app_version,
            "consent": req.consent
        }, ensure_ascii=False)
    ))

    conn.commit()
    conn.close()

    return StartSessionResponse(session_id=session_id)

@app.post("/log_event")
def log_event(req: LogEventRequest):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO events (
        session_id,
        participant_id,
        condition,
        event_type,
        client_time,
        server_time,
        scene,
        map_id,
        zone_id,
        case_id,
        sentence_id,
        source_mode,
        target_distortion,
        payload_json
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        req.session_id or "",
        req.participant_id or "",
        req.condition or "",
        req.event_type,
        float(req.client_time or 0.0),
        now_iso(),
        req.scene or "",
        req.map_id or "",
        req.zone_id or "",
        req.case_id or "",
        req.sentence_id or "",
        req.source_mode or "",
        req.target_distortion or "",
        req.payload_json or "{}"
    ))

    # 同步更新 session 的最后进度点，后面可以用来算退出点
    if req.session_id:
        cur.execute("""
        UPDATE sessions
        SET last_checkpoint = ?
        WHERE session_id = ?
        """, (
            req.event_type,
            req.session_id
        ))

    conn.commit()
    conn.close()

    return {"ok": True}

@app.post("/complete_session")
def complete_session(req: CompleteSessionRequest):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    UPDATE sessions
    SET completed = 1,
        ended_at_server = ?,
        last_checkpoint = ?
    WHERE session_id = ?
    """, (
        now_iso(),
        req.last_checkpoint or "game_complete",
        req.session_id
    ))

    cur.execute("""
    INSERT INTO events (
        session_id,
        event_type,
        client_time,
        server_time,
        payload_json
    )
    VALUES (?, ?, ?, ?, ?)
    """, (
        req.session_id,
        "game_complete",
        0.0,
        now_iso(),
        json.dumps({"last_checkpoint": req.last_checkpoint}, ensure_ascii=False)
    ))

    conn.commit()
    conn.close()

    return {"ok": True}

@app.get("/debug/events")
def debug_events(limit: int = 20):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM events
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return rows

@app.get("/debug/model_status")
def debug_model_status():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    folders = sorted([
        p for p in glob.glob(os.path.join(base_dir, "multiclass*"))
        if os.path.isdir(p)
    ])

    sample_contents = {}
    for p in folders[:5]:
        try:
            sample_contents[os.path.basename(p)] = sorted(os.listdir(p))[:30]
        except Exception as e:
            sample_contents[os.path.basename(p)] = f"error: {e}"

    info = {
        "cwd": os.getcwd(),
        "base_dir": base_dir,
        "model_root_env": os.environ.get("MODEL_ROOT", ""),
        "model_root_used": MODEL_ROOT,
        "db_path": DB_PATH,
        "multiclass_folder_count": len(folders),
        "multiclass_folders": [os.path.basename(p) for p in folders[:50]],
        "sample_contents": sample_contents,
        "files_in_base_dir": sorted(os.listdir(base_dir))[:100],
    }

    try:
        info["runtime_type"] = str(type(runtime))
    except Exception as e:
        info["runtime_type"] = f"error: {e}"

    try:
        info["runtime_model_count"] = len(runtime.models) if hasattr(runtime, "models") else "no_models_attr"
    except Exception as e:
        info["runtime_model_count"] = f"error: {e}"

    try:
        info["runtime_split_folders"] = runtime.split_folders if hasattr(runtime, "split_folders") else "no_split_folders_attr"
    except Exception as e:
        info["runtime_split_folders"] = f"error: {e}"

    return info

@app.get("/export/events.csv")
def export_events_csv():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM events
    ORDER BY id ASC
    """)

    rows_raw = [dict(r) for r in cur.fetchall()]
    conn.close()

    fieldnames = [
        "id",
        "session_id",
        "participant_id",
        "condition",
        "event_type",
        "client_time",
        "server_time",
        "scene",
        "map_id",
        "zone_id",
        "case_id",
        "sentence_id",
        "source_mode",
        "target_distortion",
        "payload_json"
    ]

    return make_csv_response("events.csv", fieldnames, rows_raw)

@app.get("/export/sessions.csv")
def export_sessions_csv():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM sessions
    ORDER BY started_at_server ASC
    """)

    rows_raw = [dict(r) for r in cur.fetchall()]
    conn.close()

    fieldnames = [
        "session_id",
        "participant_id",
        "condition",
        "platform",
        "app_version",
        "consent",
        "started_at_server",
        "ended_at_server",
        "completed",
        "last_checkpoint"
    ]

    return make_csv_response("sessions.csv", fieldnames, rows_raw)

@app.get("/export/sentence_evidence.csv")
def export_sentence_evidence_csv():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM events
    WHERE event_type = 'sentence_evidence'
    ORDER BY id ASC
    """)

    rows_raw = [dict(r) for r in cur.fetchall()]
    conn.close()

    rows = []

    for r in rows_raw:
        p = parse_payload(r.get("payload_json", ""))

        rows.append({
            "id": r.get("id"),
            "session_id": r.get("session_id"),
            "participant_id": r.get("participant_id"),
            "condition": r.get("condition"),
            "server_time": r.get("server_time"),
            "client_time": r.get("client_time"),

            "scene": r.get("scene"),
            "map_id": r.get("map_id"),
            "zone_id": r.get("zone_id"),
            "case_id": r.get("case_id"),
            "sentence_id": r.get("sentence_id"),
            "source_mode": r.get("source_mode"),
            "target_distortion": r.get("target_distortion"),

            "sentence_text": p.get("sentence_text", ""),

            "predicted_label": p.get("predicted_label", ""),
            "nlp_score": p.get("nlp_score", ""),
            "top2_label": p.get("top2_label", ""),
            "score_margin": p.get("score_margin", ""),

            "raw_keyword_count": p.get("raw_keyword_count", ""),
            "api_keyword_count": p.get("api_keyword_count", ""),
            "final_ui_keyword_count": p.get("final_ui_keyword_count", ""),
            "text_word_count": p.get("text_word_count", ""),
            "keyword_density": p.get("keyword_density", ""),

            "api_valid": p.get("api_valid", ""),
            "assessment_valid": p.get("assessment_valid", ""),
            "invalid_reason": p.get("invalid_reason", ""),

            "raw_nlp_keywords_json": json.dumps(p.get("raw_nlp_keywords", []), ensure_ascii=False),
            "final_ui_keywords_json": json.dumps(p.get("final_ui_keywords", []), ensure_ascii=False)
        })

    fieldnames = [
        "id",
        "session_id",
        "participant_id",
        "condition",
        "server_time",
        "client_time",

        "scene",
        "map_id",
        "zone_id",
        "case_id",
        "sentence_id",
        "source_mode",
        "target_distortion",

        "sentence_text",
        "predicted_label",
        "nlp_score",
        "top2_label",
        "score_margin",

        "raw_keyword_count",
        "api_keyword_count",
        "final_ui_keyword_count",
        "text_word_count",
        "keyword_density",

        "api_valid",
        "assessment_valid",
        "invalid_reason",

        "raw_nlp_keywords_json",
        "final_ui_keywords_json"
    ]

    return make_csv_response("sentence_evidence.csv", fieldnames, rows)

@app.get("/export/word_clicks.csv")
def export_word_clicks_csv():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM events
    WHERE event_type = 'word_click'
    ORDER BY id ASC
    """)

    rows_raw = [dict(r) for r in cur.fetchall()]
    conn.close()

    rows = []

    for r in rows_raw:
        p = parse_payload(r.get("payload_json", ""))

        rows.append({
            "id": r.get("id"),
            "session_id": r.get("session_id"),
            "participant_id": r.get("participant_id"),
            "condition": r.get("condition"),
            "server_time": r.get("server_time"),
            "client_time": r.get("client_time"),

            "sentence_id": r.get("sentence_id"),

            "clicked_word": p.get("clicked_word", ""),
            "word_index": p.get("word_index", ""),
            "mode": p.get("mode", ""),
            "result": p.get("result", ""),
            "time_since_sentence_start": p.get("time_since_sentence_start", ""),
            "already_collected": p.get("already_collected", "")
        })

    fieldnames = [
        "id",
        "session_id",
        "participant_id",
        "condition",
        "server_time",
        "client_time",

        "sentence_id",

        "clicked_word",
        "word_index",
        "mode",
        "result",
        "time_since_sentence_start",
        "already_collected"
    ]

    return make_csv_response("word_clicks.csv", fieldnames, rows)

@app.get("/export/session_summary.csv")
def export_session_summary_csv():
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. 读取 sessions
    cur.execute("""
    SELECT *
    FROM sessions
    ORDER BY started_at_server ASC
    """)
    sessions = [dict(r) for r in cur.fetchall()]

    # 2. 读取 events
    cur.execute("""
    SELECT *
    FROM events
    ORDER BY id ASC
    """)
    events = [dict(r) for r in cur.fetchall()]

    conn.close()

    # 3. 按 session_id 分组
    events_by_session = {}
    for e in events:
        sid = e.get("session_id") or ""
        if sid not in events_by_session:
            events_by_session[sid] = []
        events_by_session[sid].append(e)

    def safe_float(x, default=0.0):
        try:
            if x is None or x == "":
                return default
            return float(x)
        except Exception:
            return default

    def safe_int(x, default=0):
        try:
            if x is None or x == "":
                return default
            return int(float(x))
        except Exception:
            return default

    def rate(numer, denom):
        if denom <= 0:
            return ""
        return round(numer / denom, 4)

    def avg(values):
        values = [v for v in values if v is not None]
        if not values:
            return ""
        return round(sum(values) / len(values), 4)

    rows = []

    for s in sessions:
        sid = s.get("session_id", "")
        evs = events_by_session.get(sid, [])

        event_count = len(evs)
        last_event_type = evs[-1].get("event_type", "") if evs else ""

        # ---------- 基础事件计数 ----------
        sentence_evidence_events = [e for e in evs if e.get("event_type") == "sentence_evidence"]
        sentence_complete_events = [e for e in evs if e.get("event_type") == "sentence_complete"]
        word_click_events = [e for e in evs if e.get("event_type") == "word_click"]
        auto_highlight_events = [e for e in evs if e.get("event_type") == "auto_highlight"]
        zone_enter_events = [e for e in evs if e.get("event_type") == "zone_enter"]
        zone_complete_events = [e for e in evs if e.get("event_type") == "zone_complete"]
        game_complete_events = [e for e in evs if e.get("event_type") == "game_complete"]

        sentence_evidence_count = len(sentence_evidence_events)
        sentence_complete_count = len(sentence_complete_events)
        word_click_count = len(word_click_events)
        auto_highlight_count = len(auto_highlight_events)
        zone_enter_count = len(zone_enter_events)
        zone_complete_count = len(zone_complete_events)
        game_complete_count = len(game_complete_events)

        # ---------- 点击行为统计 ----------
        correct_click_count = 0
        wrong_click_count = 0

        nohint_click_count = 0
        nohint_correct_count = 0
        afterhint_click_count = 0
        afterhint_correct_count = 0
        tutorial_click_count = 0

        for e in word_click_events:
            p = parse_payload(e.get("payload_json", ""))

            result = str(p.get("result", "")).lower()
            mode = str(p.get("mode", "")).lower()

            if result == "correct":
                correct_click_count += 1
            elif result == "wrong":
                wrong_click_count += 1

            if mode == "nohint":
                nohint_click_count += 1
                if result == "correct":
                    nohint_correct_count += 1

            elif mode == "afterhint":
                afterhint_click_count += 1
                if result == "correct":
                    afterhint_correct_count += 1

            elif mode == "tutorial":
                tutorial_click_count += 1

        # ---------- NLP / sentence evidence 平均值 ----------
        nlp_scores = []
        score_margins = []
        final_keyword_counts = []
        text_word_counts = []
        keyword_densities = []

        for e in sentence_evidence_events:
            p = parse_payload(e.get("payload_json", ""))

            if p.get("nlp_score", "") != "":
                nlp_scores.append(safe_float(p.get("nlp_score")))

            if p.get("score_margin", "") != "":
                score_margins.append(safe_float(p.get("score_margin")))

            if p.get("final_ui_keyword_count", "") != "":
                final_keyword_counts.append(safe_float(p.get("final_ui_keyword_count")))

            if p.get("text_word_count", "") != "":
                text_word_counts.append(safe_float(p.get("text_word_count")))

            if p.get("keyword_density", "") != "":
                keyword_densities.append(safe_float(p.get("keyword_density")))

        total_final_ui_keyword_count = int(sum(final_keyword_counts)) if final_keyword_counts else 0

        # ---------- 完成状态 ----------
        session_completed_flag = safe_int(s.get("completed", 0))
        completed_by_game_event = 1 if game_complete_count > 0 else 0

        completed_final = 1 if (session_completed_flag == 1 or completed_by_game_event == 1) else 0

        # ---------- 用 client_time 粗略估计游玩时长 ----------
        client_times = [
            safe_float(e.get("client_time"))
            for e in evs
            if e.get("client_time") is not None
        ]

        if client_times:
            session_duration_client = round(max(client_times) - min(client_times), 3)
        else:
            session_duration_client = ""

        rows.append({
            "session_id": sid,
            "participant_id": s.get("participant_id", ""),
            "condition": s.get("condition", ""),
            "platform": s.get("platform", ""),
            "app_version": s.get("app_version", ""),
            "consent": s.get("consent", ""),

            "started_at_server": s.get("started_at_server", ""),
            "ended_at_server": s.get("ended_at_server", ""),
            "completed": completed_final,
            "last_checkpoint": s.get("last_checkpoint", ""),
            "last_event_type": last_event_type,
            "session_duration_client_seconds": session_duration_client,

            "event_count": event_count,

            "sentence_evidence_count": sentence_evidence_count,
            "sentence_complete_count": sentence_complete_count,
            "sentence_completion_rate": rate(sentence_complete_count, sentence_evidence_count),

            "word_click_count": word_click_count,
            "correct_click_count": correct_click_count,
            "wrong_click_count": wrong_click_count,
            "click_correct_rate": rate(correct_click_count, word_click_count),

            "nohint_click_count": nohint_click_count,
            "nohint_correct_count": nohint_correct_count,
            "nohint_correct_rate": rate(nohint_correct_count, nohint_click_count),

            "afterhint_click_count": afterhint_click_count,
            "afterhint_correct_count": afterhint_correct_count,
            "afterhint_correct_rate": rate(afterhint_correct_count, afterhint_click_count),

            "tutorial_click_count": tutorial_click_count,

            "auto_highlight_count": auto_highlight_count,
            "auto_highlight_per_sentence": rate(auto_highlight_count, sentence_evidence_count),

            "zone_enter_count": zone_enter_count,
            "zone_complete_count": zone_complete_count,
            "zone_completion_rate": rate(zone_complete_count, zone_enter_count),

            "game_complete_count": game_complete_count,

            "avg_nlp_score": avg(nlp_scores),
            "avg_score_margin": avg(score_margins),
            "avg_final_ui_keyword_count": avg(final_keyword_counts),
            "total_final_ui_keyword_count": total_final_ui_keyword_count,
            "avg_text_word_count": avg(text_word_counts),
            "avg_keyword_density": avg(keyword_densities)
        })

    fieldnames = [
        "session_id",
        "participant_id",
        "condition",
        "platform",
        "app_version",
        "consent",

        "started_at_server",
        "ended_at_server",
        "completed",
        "last_checkpoint",
        "last_event_type",
        "session_duration_client_seconds",

        "event_count",

        "sentence_evidence_count",
        "sentence_complete_count",
        "sentence_completion_rate",

        "word_click_count",
        "correct_click_count",
        "wrong_click_count",
        "click_correct_rate",

        "nohint_click_count",
        "nohint_correct_count",
        "nohint_correct_rate",

        "afterhint_click_count",
        "afterhint_correct_count",
        "afterhint_correct_rate",

        "tutorial_click_count",

        "auto_highlight_count",
        "auto_highlight_per_sentence",

        "zone_enter_count",
        "zone_complete_count",
        "zone_completion_rate",

        "game_complete_count",

        "avg_nlp_score",
        "avg_score_margin",
        "avg_final_ui_keyword_count",
        "total_final_ui_keyword_count",
        "avg_text_word_count",
        "avg_keyword_density"
    ]

    return make_csv_response("session_summary.csv", fieldnames, rows)