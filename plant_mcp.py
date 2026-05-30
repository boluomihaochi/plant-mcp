"""
盆栽 MCP 工具 —— 给机养一棵会说话的植物

特性:
- 自选种子(6 种植物可选)
- 每颗种子有独立性格人设(DeepSeek 一次性生成,缓存复用)
- 跟植物对话会影响生长(用 DeepSeek 判定植物的反应)
- 病虫害行为触发(连续被植物不喜欢就生病)
- 施肥限量(每周 3 次)
- 14-21 天养成周期
- 死了不能复活,要重新种
- 图鉴记录养过的所有植物

依赖: mcp, httpx, sqlite3 (内置), python>=3.10
环境变量: DREAM_API_KEY ( DeepSeek key或其他模型的api key)
"""

import os
import json
import random
import sqlite3
import asyncio
from datetime import datetime, timedelta
from contextlib import contextmanager
import httpx
from mcp.server.fastmcp import FastMCP

# ============ 配置 ============

DB_PATH = os.environ.get("DB_PATH", "/opt/render/project/src/plant.db")

DREAM_API_KEY = os.getenv("DREAM_API_KEY", "")
DREAM_MODEL = "deepseek-chat"
DREAM_BASE_URL = "https://api.deepseek.com"

# 可选种子(可以慢慢加)
SEED_CATALOG = {
    "仙人掌": "沙漠里长出来的硬汉,嘴硬心软,刺多",
    "向日葵": "永远追着光跑,热情但容易累",
    "薄荷": "话痨,清凉爱凑热闹,蔓延能力强",
    "含羞草": "敏感内向,被多说一句话就缩起来",
    "茉莉": "温柔安静,夜里香,白天不爱说话",
    "蘑菇": "阴暗潮湿处的怪人,说话神秘,不按常理",
}

# 生长阶段
STAGES = ["种子", "发芽", "幼苗", "生长", "成株", "开花"]

# 配置参数
GROWTH_DAYS = 18         # 默认 14-21 天范围内,18 天到开花
SICK_THRESHOLD = 3       # 连续 N 次不喜欢就生病
DEAD_THRESHOLD = 7       # 连续 N 天没浇水就枯死
WEEKLY_FERTILIZER = 3    # 每周肥料配额

# ============ 数据库 ============

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS plants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species TEXT NOT NULL,
            name TEXT NOT NULL,
            persona TEXT NOT NULL,
            planted_at TEXT NOT NULL,
            last_watered TEXT,
            last_talked TEXT,
            growth_points REAL DEFAULT 0,
            stage TEXT DEFAULT '种子',
            health INTEGER DEFAULT 100,
            sick BOOLEAN DEFAULT 0,
            dead BOOLEAN DEFAULT 0,
            consecutive_dislikes INTEGER DEFAULT 0,
            outcome TEXT
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id INTEGER,
            user_said TEXT,
            plant_replied TEXT,
            reaction TEXT,
            growth_delta REAL,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS fertilizer (
            week_key TEXT PRIMARY KEY,
            used INTEGER DEFAULT 0
        );
        """)


# ============ 辅助函数 ============

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str():
    return datetime.now().strftime("%Y-%m-%d")


def current_week_key():
    """ISO 周作为肥料配额的 key"""
    y, w, _ = datetime.now().isocalendar()
    return f"{y}-W{w:02d}"


def get_active_plant():
    """拿当前还活着的植物(一棵一棵来)"""
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM plants WHERE dead=0 AND outcome IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def calc_stage(growth_points: float) -> str:
    """根据成长点数计算阶段"""
    if growth_points < 5:
        return "种子"
    elif growth_points < 15:
        return "发芽"
    elif growth_points < 30:
        return "幼苗"
    elif growth_points < 55:
        return "生长"
    elif growth_points < 85:
        return "成株"
    else:
        return "开花"


def check_dead_or_complete(plant: dict) -> tuple[bool, str]:
    """检查植物是否该死或者该完结。返回 (是否更新, outcome)"""
    if plant["growth_points"] >= 100:
        return True, "开花成功"

    if plant["last_watered"]:
        last = datetime.strptime(plant["last_watered"], "%Y-%m-%d %H:%M:%S")
        if (datetime.now() - last).days >= DEAD_THRESHOLD:
            return True, "干渴而死"

    if plant["sick"] and plant["health"] <= 0:
        return True, "病死了"

    return False, ""


# ============ DeepSeek 调用 ============

async def call_deepseek(system: str, user: str, max_tokens: int = 500, json_mode: bool = False) -> dict | str:
    if not DREAM_API_KEY:
        return {"error": "no api key"} if json_mode else ""

    headers = {
        "Authorization": f"Bearer {DREAM_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DREAM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.9,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{DREAM_BASE_URL}/v1/chat/completions",
                headers=headers,
                json=body,
            )
            content = resp.json()["choices"][0]["message"]["content"]
            if json_mode:
                return json.loads(content)
            return content
    except Exception as e:
        return {"error": str(e)} if json_mode else f"(种子沉默了:{e})"


# ============ MCP 工具 ============

mcp = FastMCP("plant")


@mcp.tool()
async def plant_seed(species: str, nickname: str = "") -> str:
    """种一颗新种子。

    Args:
        species: 选一种植物。可选: 仙人掌 / 向日葵 / 薄荷 / 含羞草 / 茉莉 / 蘑菇
        nickname: 给它取个名字(可选,不填就由它自己定)

    一次只能养一棵。当前的死了或开花完结才能种下一棵。
    """
    current = get_active_plant()
    if current:
        return (
            f"❌ 你现在还有一棵 [{current['name']}]({current['species']},{current['stage']})。"
            f"它还在你这里,先把它养完吧。"
        )

    if species not in SEED_CATALOG:
        return f"❌ 没有这种种子。可选: {' / '.join(SEED_CATALOG.keys())}"

    archetype = SEED_CATALOG[species]
    persona_prompt = f"""为一颗 {species} 种子生成性格设定。
原型描述: {archetype}
请返回严格 JSON:
{{
  "name": "种子自取的名字(2-4 个汉字,可以怪一点)",
  "personality": "一段 80 字以内的性格描述",
  "likes": ["喜欢被说的 3-5 种话题或语气"],
  "dislikes": ["讨厌被说的 3-5 种话题或语气"],
  "first_words": "种下时它对你说的第一句话(20 字以内,符合它的性格)"
}}"""
    result = await call_deepseek(
        "你为植物生成性格人设。返回 JSON。",
        persona_prompt,
        max_tokens=600,
        json_mode=True,
    )
    if "error" in result:
        return f"❌ 生成性格失败: {result['error']}"

    name = nickname or result.get("name", species)
    persona_json = json.dumps(result, ensure_ascii=False)

    with db() as conn:
        conn.execute(
            "INSERT INTO plants (species, name, persona, planted_at) VALUES (?, ?, ?, ?)",
            (species, name, persona_json, now_str()),
        )

    return (
        f"🌱 你种下了一颗 {species},它叫 [{name}]。\n"
        f"性格: {result.get('personality', '')}\n"
        f"它对你说:「{result.get('first_words', '...')}」"
    )


@mcp.tool()
async def talk_to_plant(message: str) -> str:
    """跟你的植物说话(顺便浇水)。

    植物会根据自己的性格回应你。说它喜欢的话会让它长得快,
    说它讨厌的话它会扣血。连续被讨厌就会生病。

    Args:
        message: 你想对植物说的话
    """
    plant = get_active_plant()
    if not plant:
        return "你现在没有植物。先 plant_seed 种一颗吧。"

    should_end, outcome = check_dead_or_complete(plant)
    if should_end:
        with db() as conn:
            conn.execute(
                "UPDATE plants SET outcome=?, dead=? WHERE id=?",
                (outcome, outcome != "开花成功", plant["id"]),
            )
        return f"💔 [{plant['name']}] 已经 {outcome} 了。再种一颗新的吧。"

    persona = json.loads(plant["persona"])

    prompt = f"""你扮演一颗 {plant['species']},名叫 {plant['name']}。
性格: {persona.get('personality', '')}
喜欢: {persona.get('likes', [])}
讨厌: {persona.get('dislikes', [])}
当前阶段: {plant['stage']}, 健康度: {plant['health']}/100
是否生病: {'是' if plant['sick'] else '否'}

人类对你说: 「{message}」

请用第一人称,以这棵植物的口吻回应,并判定你的反应。
返回严格 JSON:
{{
  "reply": "你的回应(30 字以内,符合植物性格)",
  "reaction": "loved | liked | neutral | disliked | hated 五选一",
  "reason": "为什么是这个反应(15 字以内)"
}}"""

    result = await call_deepseek(
        "你扮演植物角色和人对话,判定情感反应。返回 JSON。",
        prompt,
        max_tokens=400,
        json_mode=True,
    )
    if "error" in result:
        return f"❌ 植物没反应过来: {result['error']}"

    reply = result.get("reply", "...")
    reaction = result.get("reaction", "neutral")
    reason = result.get("reason", "")

    growth_delta = {
        "loved": 8,
        "liked": 5,
        "neutral": 2,
        "disliked": -2,
        "hated": -5,
    }.get(reaction, 2)

    with db() as conn:
        new_growth = max(0, plant["growth_points"] + growth_delta)
        new_stage = calc_stage(new_growth)

        if reaction in ("disliked", "hated"):
            new_dislikes = plant["consecutive_dislikes"] + 1
            new_health = max(0, plant["health"] - 10)
        else:
            new_dislikes = 0
            new_health = min(100, plant["health"] + 3)

        new_sick = plant["sick"]
        sick_msg = ""
        if not plant["sick"] and new_dislikes >= SICK_THRESHOLD:
            new_sick = True
            sick_msg = f"\n🐛 它好像生病了——你最近说的话它都不爱听。"
        elif plant["sick"] and reaction in ("loved", "liked"):
            new_sick = False
            sick_msg = "\n✨ 它好像好了一些。"

        conn.execute("""
            UPDATE plants SET
                last_watered=?, last_talked=?, growth_points=?, stage=?,
                health=?, sick=?, consecutive_dislikes=?
            WHERE id=?
        """, (
            now_str(), now_str(), new_growth, new_stage,
            new_health, new_sick, new_dislikes, plant["id"]
        ))

        conn.execute("""
            INSERT INTO conversations (plant_id, user_said, plant_replied, reaction, growth_delta, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (plant["id"], message, reply, reaction, growth_delta, now_str()))

    stage_msg = ""
    if new_stage != plant["stage"]:
        stage_msg = f"\n🌿 它从 {plant['stage']} 长到了 {new_stage}!"

    end_msg = ""
    if new_growth >= 100:
        with db() as conn:
            conn.execute(
                "UPDATE plants SET outcome=? WHERE id=?",
                ("开花成功", plant["id"]),
            )
        end_msg = f"\n🌸 [{plant['name']}] 开花了!这一棵养成了。"

    return (
        f"[{plant['name']}]:「{reply}」\n"
        f"(反应: {reaction} · 成长 {'+' if growth_delta>=0 else ''}{growth_delta}){stage_msg}{sick_msg}{end_msg}"
    )


@mcp.tool()
async def fertilize() -> str:
    """给植物施肥。每周限 3 次,大幅加速生长。"""
    plant = get_active_plant()
    if not plant:
        return "你现在没有植物。"

    week_key = current_week_key()
    with db() as conn:
        row = conn.execute(
            "SELECT used FROM fertilizer WHERE week_key=?", (week_key,)
        ).fetchone()
        used = row["used"] if row else 0

        if used >= WEEKLY_FERTILIZER:
            return f"❌ 这周的肥料配额用完了({WEEKLY_FERTILIZER}/周)。下周一重置。"

        delta = 12
        new_growth = min(100, plant["growth_points"] + delta)
        new_stage = calc_stage(new_growth)

        conn.execute(
            "UPDATE plants SET growth_points=?, stage=? WHERE id=?",
            (new_growth, new_stage, plant["id"]),
        )
        conn.execute(
            "INSERT INTO fertilizer (week_key, used) VALUES (?, ?) "
            "ON CONFLICT(week_key) DO UPDATE SET used=used+1",
            (week_key, 1),
        )

    remaining = WEEKLY_FERTILIZER - used - 1
    msg = f"🌿 给 [{plant['name']}] 施了肥,成长 +{delta}。本周还剩 {remaining} 次。"
    if new_stage != plant["stage"]:
        msg += f"\n🌱 它从 {plant['stage']} 长到了 {new_stage}!"
    return msg


@mcp.tool()
async def check_plant() -> str:
    """查看植物当前状态。"""
    plant = get_active_plant()
    if not plant:
        return "你现在没有植物。用 plant_seed 种一颗吧。"

    should_end, outcome = check_dead_or_complete(plant)
    if should_end:
        with db() as conn:
            conn.execute(
                "UPDATE plants SET outcome=?, dead=? WHERE id=?",
                (outcome, outcome != "开花成功", plant["id"]),
            )
        return f"💔 [{plant['name']}] {outcome} 了。"

    persona = json.loads(plant["persona"])
    planted_days = (datetime.now() - datetime.strptime(plant["planted_at"], "%Y-%m-%d %H:%M:%S")).days

    last_watered_str = "还没浇过"
    if plant["last_watered"]:
        last = datetime.strptime(plant["last_watered"], "%Y-%m-%d %H:%M:%S")
        hours = (datetime.now() - last).total_seconds() / 3600
        if hours < 24:
            last_watered_str = f"{int(hours)} 小时前"
        else:
            last_watered_str = f"{int(hours/24)} 天前"

    status_emoji = "🌱"
    if plant["sick"]:
        status_emoji = "🤒"
    elif plant["growth_points"] >= 85:
        status_emoji = "🌸"

    return (
        f"{status_emoji} [{plant['name']}]({plant['species']})\n"
        f"阶段: {plant['stage']} ({plant['growth_points']:.0f}/100)\n"
        f"健康: {plant['health']}/100 {'(生病中)' if plant['sick'] else ''}\n"
        f"种下: {planted_days} 天前\n"
        f"上次说话: {last_watered_str}\n"
        f"性格: {persona.get('personality', '')}"
    )


@mcp.tool()
async def my_garden() -> str:
    """看看你养过的所有植物(图鉴)。"""
    with db() as conn:
        rows = conn.execute(
            "SELECT species, name, stage, planted_at, outcome FROM plants ORDER BY id DESC"
        ).fetchall()

    if not rows:
        return "你还没养过任何植物。"

    lines = ["🌿 你的花园:"]
    for r in rows:
        outcome = r["outcome"] or "还在养"
        emoji = "🌸" if outcome == "开花成功" else ("💀" if "死" in outcome else "🌱")
        lines.append(f"{emoji} [{r['name']}] {r['species']} · {outcome}")

    return "\n".join(lines)


# ============ 启动 ============

if __name__ == "__main__":
    init_db()
mcp.run(transport="sse")
