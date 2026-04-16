import asyncio
import hashlib
import json
import random
import re
import sqlite3
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, MessageEventResult, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.star import Star, StarTools, register
from astrbot.core.provider.provider import Provider
from astrbot.core.star.filter.command import GreedyStr


DEFAULT_REMINDER_TEMPLATES = [
    "{name}，别摸了，`{task}` 还挂着。先去把这些做掉：{todos}。{remaining}",
    "{name}，现在不是装死的时候。`{task}` 还没完，先处理：{todos}。{remaining}",
    "{name}，机器人已经盯上你了。`{task}` 还在待办里：{todos}。{remaining}",
    "{name}，该开工了。今天先把 `{task}` 往前推，别再拖。重点：{todos}。{remaining}",
]

GENERIC_LLM_REPLY_NEEDLES = (
    "ready to help with the task",
    "available tools to make progress",
    "可以使用可用工具推进进度",
    "我已准备好帮助完成任务",
)

SKIP_COMMAND_PREFIXES = (
    "监督",
    "/监督",
    "督工",
    "/督工",
    "更新内容",
    "/更新内容",
    "内容预告",
    "/内容预告",
)

DEFAULT_NORMAL_CHAT_YIELD_PREFIXES = ("/", "／")

@register(
    "WorkSupervisor",
    "Codex",
    "带冷却监督、群聊@目标、每日更新/预告播报、支持大模型人格催促的 AstrBot 插件",
    "0.1.10",
    "https://github.com/AstrBotDevs/AstrBot",
)
class WorkSupervisorPlugin(Star):
    # 文件主流程：
    # 1. 通过命令创建/查询/完成/取消监督任务。
    # 2. 监听普通消息，在目标用户发言时按冷却规则触发提醒。
    # 3. 通过每日固定时间任务发送“更新内容”和“内容预告”。
    # 4. 提醒文案优先走 LLM，失败时回退到本地模板。
    def __init__(self, context: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}
        self.data_dir = Path(str(StarTools.get_data_dir()))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "work_supervisor.db"
        self._db_lock = asyncio.Lock()
        self._scheduler_task: asyncio.Task[None] | None = None
        self._inflight_task_ids: set[int] = set()
        self._last_settings_tasks_signature = ""
        self._last_broadcast_settings_signature = ""
        self._init_db()
        self._sync_settings_tasks_from_config_sync(force=True)
        self._bootstrap_broadcast_jobs_sync()

    # ---- 配置解析 ----
    def _get_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        return self._get_bool_from_value(value, default)

    def _get_bool_from_value(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _get_int(self, key: str, default: int, minimum: int = 0) -> int:
        try:
            value = int(float(self.config.get(key, default)))
        except (TypeError, ValueError):
            value = default
        return max(value, minimum)

    def _get_text(self, key: str, default: str = "") -> str:
        value = str(self.config.get(key, default) or "").strip()
        return value or default

    def _parse_text_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value or "").strip()
        if not text:
            return []
        normalized = (
            text.replace("\r", "\n")
            .replace("，", "\n")
            .replace(",", "\n")
            .replace("；", "\n")
            .replace(";", "\n")
        )
        return [line.strip() for line in normalized.split("\n") if line.strip()]

    def _parse_multiline_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value or "").strip()
        if not text:
            return []
        return [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]

    def _enabled(self) -> bool:
        return self._get_bool("enabled", True)

    def _ignore_self_messages(self) -> bool:
        return self._get_bool("ignore_self_messages", True)

    def _allow_private_chat(self) -> bool:
        return self._get_bool("allow_private_chat", True)

    def _allow_supervise_others(self) -> bool:
        return self._get_bool("allow_supervise_others", True)

    def _allow_non_admin_supervise_others(self) -> bool:
        return self._get_bool("allow_non_admin_supervise_others", False)

    def _default_duration_minutes(self) -> int:
        return self._get_int("default_duration_minutes", 180, minimum=1)

    def _default_cooldown_minutes(self) -> int:
        return self._get_int("default_cooldown_minutes", 120, minimum=0)

    def _default_todo_pick_count(self) -> int:
        return self._get_int("default_todo_pick_count", 3, minimum=1)

    def _max_todo_items_per_task(self) -> int:
        return self._get_int("max_todo_items_per_task", 12, minimum=1)

    def _allowed_group_ids(self) -> list[str]:
        return self._parse_text_list(self.config.get("allowed_group_ids", ""))

    def _blocked_group_ids(self) -> list[str]:
        return self._parse_text_list(self.config.get("blocked_group_ids", ""))

    def _scheduler_tick_seconds(self) -> int:
        return self._get_int("scheduler_tick_seconds", 20, minimum=10)

    def _llm_enabled(self) -> bool:
        return self._get_bool("llm_enabled", True)

    def _llm_provider_id(self) -> str:
        return self._get_text("llm_provider_id", "")

    def _llm_follow_active_persona(self) -> bool:
        return self._get_bool("llm_follow_active_persona", True)

    def _llm_custom_prompt(self) -> str:
        return self._get_text("llm_custom_prompt", "")

    def _llm_max_output_chars(self) -> int:
        return self._get_int("llm_max_output_chars", 180, minimum=60)

    def _update_feature_enabled(self) -> bool:
        return self._get_bool("update_feature_enabled", True)

    def _preview_feature_enabled(self) -> bool:
        return self._get_bool("preview_feature_enabled", True)

    def _fallback_templates(self) -> list[str]:
        configured = self._parse_multiline_list(
            self.config.get("fallback_reminder_templates", "")
        )
        return configured or DEFAULT_REMINDER_TEMPLATES

    def _normal_chat_yield_prefixes(self) -> list[str]:
        configured = self._parse_multiline_list(
            self.config.get("normal_chat_yield_prefixes", "")
        )
        cleaned = [item for item in configured if item]
        return cleaned or list(DEFAULT_NORMAL_CHAT_YIELD_PREFIXES)

    def _settings_tasks_config(self) -> list[dict[str, Any]]:
        value = self.config.get("settings_tasks", [])
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    def _broadcast_settings_config(self) -> list[dict[str, Any]]:
        value = self.config.get("broadcast_settings", [])
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    def _settings_tasks_signature(self) -> str:
        return json.dumps(
            self._settings_tasks_config(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _broadcast_settings_signature(self) -> str:
        return json.dumps(
            self._broadcast_settings_config(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _save_plugin_config(self, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            self.config[key] = value
        save_config = getattr(self.config, "save_config", None)
        if callable(save_config):
            save_config()

    # ---- 数据库 ----
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS supervision_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_user_id TEXT NOT NULL,
                    target_user_name TEXT NOT NULL,
                    trigger_session_id TEXT NOT NULL,
                    trigger_group_id TEXT DEFAULT '',
                    created_by_user_id TEXT NOT NULL,
                    created_by_user_name TEXT NOT NULL,
                    task_title TEXT NOT NULL,
                    todo_items_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    schedule_kind TEXT NOT NULL DEFAULT 'once',
                    cooldown_seconds INTEGER NOT NULL,
                    reminder_limit INTEGER NOT NULL DEFAULT 0,
                    todo_pick_count INTEGER NOT NULL,
                    last_reminded_at TEXT DEFAULT '',
                    completed_at TEXT DEFAULT '',
                    cancelled_at TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    settings_task_key TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_supervision_tasks_target_status
                    ON supervision_tasks(target_user_id, status);

                CREATE TABLE IF NOT EXISTS reminder_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    target_user_id TEXT NOT NULL,
                    trigger_session_id TEXT NOT NULL,
                    reminder_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_reminder_logs_task
                    ON reminder_logs(task_id, created_at);

                CREATE TABLE IF NOT EXISTS broadcast_jobs (
                    kind TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    session_id TEXT NOT NULL DEFAULT '',
                    session_label TEXT NOT NULL DEFAULT '',
                    time_hhmm TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    last_sent_at TEXT NOT NULL DEFAULT '',
                    updated_by_user_id TEXT NOT NULL DEFAULT '',
                    updated_by_user_name TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT ''
                );
                """
            )
            self._ensure_column(conn, "supervision_tasks", "settings_task_key", "TEXT DEFAULT ''")
            self._ensure_column(conn, "supervision_tasks", "schedule_kind", "TEXT NOT NULL DEFAULT 'once'")
            self._ensure_column(conn, "supervision_tasks", "reminder_limit", "INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_supervision_tasks_settings_key
                    ON supervision_tasks(settings_task_key, status)
                """
            )
            conn.commit()

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_sql: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in columns:
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

    # ---- 时间与格式 ----
    def _now(self) -> datetime:
        return datetime.now().astimezone()

    def _iso(self, value: datetime) -> str:
        return value.isoformat()

    def _parse_dt(self, value: str | None) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _format_time(self, value: datetime | None) -> str:
        if value is None:
            return "未设置"
        return value.strftime("%Y-%m-%d %H:%M")

    def _format_remaining(self, end_at: datetime, now: datetime) -> str:
        if end_at <= now:
            return "已经到结束时间了。"
        delta = end_at - now
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"剩余时间大约 {hours} 小时 {minutes} 分钟。"
        if minutes > 0:
            return f"剩余时间大约 {minutes} 分钟。"
        return "剩余时间不到 1 分钟。"

    def _format_until_start(self, start_at: datetime, now: datetime) -> str:
        if start_at <= now:
            return "已经到开始时间了。"
        delta = start_at - now
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"距离开始大约还有 {hours} 小时 {minutes} 分钟。"
        if minutes > 0:
            return f"距离开始大约还有 {minutes} 分钟。"
        return "距离开始不到 1 分钟。"

    def _format_duration_seconds(self, seconds: int) -> str:
        seconds = max(int(seconds), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours and minutes:
            return f"{hours} 小时 {minutes} 分钟"
        if hours:
            return f"{hours} 小时"
        if minutes:
            return f"{minutes} 分钟"
        return f"{seconds} 秒"

    def _format_compact_duration_seconds(self, seconds: int) -> str:
        seconds = max(int(seconds), 0)
        if seconds and seconds % 86400 == 0:
            return f"{seconds // 86400}d"
        if seconds and seconds % 3600 == 0:
            return f"{seconds // 3600}h"
        if seconds and seconds % 60 == 0:
            return f"{seconds // 60}m"
        return f"{seconds}s"

    def _normalize_schedule_kind(self, value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"permanent", "forever", "always", "永久"}:
            return "permanent"
        if raw in {"daily", "everyday", "每天", "每日"}:
            return "daily"
        return "once"

    def _parse_duration_seconds_or_none(self, text: str) -> int | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        parsed = self._parse_duration_seconds(raw, -1)
        return parsed if parsed > 0 else None

    def _parse_settings_duration_spec(self, text: Any) -> tuple[str, int | None, str]:
        raw = str(text or "").strip()
        lowered = raw.lower()
        if not raw:
            return "once", None, ""
        if lowered in {"永久", "permanent", "forever", "always"}:
            return "permanent", None, "永久"
        if lowered in {"每天", "daily", "everyday"}:
            return "daily", 86400, "每天"
        duration_seconds = self._parse_duration_seconds_or_none(raw)
        if duration_seconds is None:
            return "once", None, raw
        return "once", duration_seconds, self._format_compact_duration_seconds(duration_seconds)

    def _format_schedule_duration_label(self, task: dict[str, Any]) -> str:
        schedule_kind = self._normalize_schedule_kind(task.get("schedule_kind"))
        if schedule_kind == "permanent":
            return "永久"
        if schedule_kind == "daily":
            return "每天"
        start_at = self._parse_dt(task.get("start_at"))
        end_at = self._parse_dt(task.get("end_at"))
        if start_at and end_at and end_at > start_at:
            return self._format_duration_seconds(int((end_at - start_at).total_seconds()))
        return "未设置"

    def _task_cycle_start(self, task: dict[str, Any], now: datetime) -> datetime | None:
        start_at = self._parse_dt(task.get("start_at"))
        if start_at is None:
            return None
        if self._normalize_schedule_kind(task.get("schedule_kind")) != "daily":
            return start_at
        if now <= start_at:
            return start_at
        cycle_seconds = 86400
        elapsed = max(int((now - start_at).total_seconds()), 0)
        cycles = elapsed // cycle_seconds
        return start_at + timedelta(days=cycles)

    def _task_cycle_end(self, task: dict[str, Any], now: datetime) -> datetime | None:
        schedule_kind = self._normalize_schedule_kind(task.get("schedule_kind"))
        if schedule_kind == "permanent":
            return None
        if schedule_kind == "daily":
            cycle_start = self._task_cycle_start(task, now)
            if cycle_start is None:
                return None
            return cycle_start + timedelta(days=1)
        return self._parse_dt(task.get("end_at"))

    def _format_reminder_limit_text(self, limit: int) -> str:
        if limit <= 0:
            return "不限制"
        return f"最多 {limit} 次"

    def _parse_settings_datetime(self, value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text.replace("/", "-")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(normalized, fmt).astimezone()
            except ValueError:
                continue
        return self._parse_dt(normalized)

    def _normalize_hhmm(self, text: str) -> str | None:
        raw = str(text or "").strip()
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return f"{hour:02d}:{minute:02d}"

    # ---- 事件辅助 ----
    def _is_private_chat(self, event: AstrMessageEvent) -> bool:
        try:
            return bool(event.is_private_chat())
        except Exception:
            group_id = str(event.get_group_id() or "").strip()
            return not bool(group_id)

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        try:
            return bool(event.is_admin())
        except Exception:
            return False

    def _extract_sender_name(self, event: AstrMessageEvent) -> str:
        for attr in ("get_sender_name", "get_sender_nickname"):
            try:
                value = getattr(event, attr)()
                if value:
                    return str(value)
            except Exception:
                continue
        return str(event.get_sender_id() or "未知用户")

    def _extract_plain_text(self, event: AstrMessageEvent) -> str:
        plain_parts: list[str] = []
        for component in event.get_messages():
            if isinstance(component, Plain):
                plain_parts.append(component.text)
        merged = "".join(plain_parts).strip()
        return merged or str(event.get_message_str() or "").strip()

    def _extract_mentions(self, event: AstrMessageEvent) -> list[dict[str, str]]:
        mentions: list[dict[str, str]] = []
        self_id = str(event.get_self_id() or "").strip()
        for component in event.get_messages():
            if not isinstance(component, At):
                continue
            qq = str(getattr(component, "qq", "") or "").strip()
            if not qq or qq == "all" or qq == self_id:
                continue
            mentions.append(
                {
                    "user_id": qq,
                    "name": str(getattr(component, "name", "") or qq).strip() or qq,
                }
            )
        return mentions

    def _event_group_allowed(self, event: AstrMessageEvent) -> bool:
        if self._is_private_chat(event):
            return True
        group_id = str(event.get_group_id() or "").strip()
        if group_id in self._blocked_group_ids():
            return False
        allowed = self._allowed_group_ids()
        if not allowed:
            return True
        return group_id in allowed

    def _command_gate_error(self, event: AstrMessageEvent) -> str | None:
        if not self._enabled():
            return "监督插件当前已关闭。"
        if self._ignore_self_messages() and event.get_sender_id() == event.get_self_id():
            return "__ignore__"
        if not self._allow_private_chat() and self._is_private_chat(event):
            return "当前配置不允许在私聊里使用这个插件。"
        if not self._event_group_allowed(event):
            return "当前群聊不在插件允许范围内。"
        return None

    def _skip_message_for_commands(self, text: str) -> bool:
        normalized = self._normalize_command_text(text)
        return any(normalized.startswith(prefix) for prefix in SKIP_COMMAND_PREFIXES)

    def _message_mentions_self(self, event: AstrMessageEvent) -> bool:
        self_id = str(event.get_self_id() or "").strip()
        if not self_id:
            return False
        for component in event.get_messages():
            if not isinstance(component, At):
                continue
            qq = str(getattr(component, "qq", "") or "").strip()
            if qq and qq == self_id:
                return True
        return False

    def _should_yield_to_normal_chat(
        self,
        event: AstrMessageEvent,
        plain_text: str,
    ) -> bool:
        # Passive supervision must not swallow explicit bot wake-ups. When the
        # target is intentionally talking to AstrBot via @ or wake-prefix
        # commands, leave the message to the normal chat/command pipeline.
        if bool(getattr(event, "is_at_or_wake_command", False)):
            return True
        if self._message_mentions_self(event):
            return True
        normalized = self._normalize_command_text(plain_text)
        return any(
            normalized.startswith(prefix)
            for prefix in self._normal_chat_yield_prefixes()
        )

    def _normalize_command_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    def _strip_command_prefix(
        self,
        text: str,
        *command_names: str,
    ) -> tuple[str, bool]:
        normalized = self._normalize_command_text(text)
        candidates: list[str] = []
        for command_name in command_names:
            candidate = self._normalize_command_text(command_name)
            if not candidate:
                continue
            candidates.append(candidate)
            if not candidate.startswith("/"):
                candidates.append(f"/{candidate}")

        for candidate in candidates:
            if normalized == candidate:
                return "", True
            if normalized.startswith(f"{candidate} "):
                return normalized[len(candidate) :].strip(), True
        return normalized, False

    def _resolve_command_payload(
        self,
        event: AstrMessageEvent,
        payload: str,
        *command_names: str,
    ) -> str:
        # AstrBot 4.23.1 currently resolves `GreedyStr = ""` as a normal string
        # parameter, so command handlers only receive the first token. Rebuild the
        # payload from raw message text first and fall back to parsed params only
        # when the command prefix cannot be matched from the original event text.
        full_text = self._extract_plain_text(event)
        stripped_text, matched = self._strip_command_prefix(full_text, *command_names)
        if matched:
            return stripped_text
        return str(payload or "").strip()

    def _supervision_help_text(self) -> str:
        return (
            "监督命令：\n"
            "监督 开始 任务=写第一章 待办=大纲、正文、校对 时长=3h 冷却=2h 抽取=3\n"
            "监督 开始 @小明 任务=做海报 待办=出图、排版 时长=2h 冷却=1h\n"
            "监督 状态 [@目标]\n"
            "监督 完成 [@目标]\n"
            "监督 取消 [@目标]\n\n"
            "参数说明：任务=必填；待办=可选，用顿号或逗号分隔；时长=默认配置值；冷却=提醒间隔；抽取=每次提醒展示几条待办。\n\n"
            "播报命令：\n"
            "更新内容 设置 时间=21:00 内容=今天更新了第一章和封面\n"
            "内容预告 设置 时间=20:00 内容=明天预告第二章和设定图\n"
            "更新内容 状态 / 更新内容 开 / 更新内容 关 / 更新内容 立即发送\n"
            "内容预告 状态 / 内容预告 开 / 内容预告 关 / 内容预告 立即发送\n\n"
            "提醒逻辑：目标用户在原会话中发言时，如果超过提醒间隔，就会触发一次监督提醒。"
        )

    def _strip_explicit_command_root(
        self,
        text: str,
        *roots: str,
    ) -> str | None:
        normalized = self._normalize_command_text(text)
        for root in roots:
            candidate = self._normalize_command_text(root)
            if not candidate:
                continue
            if normalized == candidate:
                return ""
            if normalized.startswith(f"{candidate} "):
                return normalized[len(candidate) :].strip()
        return None

    async def _emit_direct_command_result(
        self,
        event: AstrMessageEvent,
        command_call,
        *args: Any,
    ) -> bool:
        last_result: MessageEventResult | None = None
        async for item in command_call(event, *args):
            last_result = item
        if last_result is None:
            return False
        event.set_result(last_result.stop_event())
        return True

    async def _maybe_handle_explicit_command(self, event: AstrMessageEvent) -> bool:
        # Keep an explicit command fallback in the message hook so plugin commands
        # still work when AstrBot wake-prefix parsing leaves handlers with empty or
        # truncated payloads.
        normalized_text = self._normalize_command_text(self._extract_plain_text(event))
        if not normalized_text:
            return False

        supervision_tail = self._strip_explicit_command_root(normalized_text, "监督", "督工")
        if supervision_tail is not None:
            if not supervision_tail:
                return await self._emit_direct_command_result(event, self.help_supervision)
            subcommand, _, remainder = supervision_tail.partition(" ")
            if subcommand in {"开始", "创建", "新增"}:
                return await self._emit_direct_command_result(
                    event,
                    self.start_supervision,
                    remainder.strip(),
                )
            if subcommand in {"状态", "查看", "查询"}:
                return await self._emit_direct_command_result(event, self.status_supervision)
            if subcommand in {"完成", "结束", "done"}:
                return await self._emit_direct_command_result(event, self.complete_supervision)
            if subcommand in {"取消", "abort"}:
                return await self._emit_direct_command_result(event, self.cancel_supervision)
            if subcommand in {"帮助", "help"}:
                return await self._emit_direct_command_result(event, self.help_supervision)
            event.set_result(MessageEventResult().message(self._supervision_help_text()).stop_event())
            return True

        update_tail = self._strip_explicit_command_root(normalized_text, "更新内容")
        if update_tail is not None:
            if not update_tail:
                return await self._emit_direct_command_result(event, self.update_status)
            subcommand, _, remainder = update_tail.partition(" ")
            if subcommand in {"设置", "设定"}:
                return await self._emit_direct_command_result(event, self.update_set, remainder.strip())
            if subcommand == "开":
                return await self._emit_direct_command_result(event, self.update_on)
            if subcommand == "关":
                return await self._emit_direct_command_result(event, self.update_off)
            if subcommand == "状态":
                return await self._emit_direct_command_result(event, self.update_status)
            if subcommand in {"立即发送", "发送"}:
                return await self._emit_direct_command_result(
                    event,
                    self.update_send_now,
                    remainder.strip(),
                )
            return await self._emit_direct_command_result(event, self.update_status)

        preview_tail = self._strip_explicit_command_root(normalized_text, "内容预告")
        if preview_tail is not None:
            if not preview_tail:
                return await self._emit_direct_command_result(event, self.preview_status)
            subcommand, _, remainder = preview_tail.partition(" ")
            if subcommand in {"设置", "设定"}:
                return await self._emit_direct_command_result(event, self.preview_set, remainder.strip())
            if subcommand == "开":
                return await self._emit_direct_command_result(event, self.preview_on)
            if subcommand == "关":
                return await self._emit_direct_command_result(event, self.preview_off)
            if subcommand == "状态":
                return await self._emit_direct_command_result(event, self.preview_status)
            if subcommand in {"立即发送", "发送"}:
                return await self._emit_direct_command_result(
                    event,
                    self.preview_send_now,
                    remainder.strip(),
                )
            return await self._emit_direct_command_result(event, self.preview_status)

        return False

    # ---- 调度器 ----
    async def _ensure_scheduler_started(self) -> None:
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self._sync_settings_tasks_from_config()
                await self._sync_broadcast_jobs_from_config()
                await self._expire_overdue_tasks()
                await self._run_due_settings_initial_reminders()
                await self._run_due_broadcasts()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"WorkSupervisor scheduler loop failed: {exc}", exc_info=True)
            await asyncio.sleep(self._scheduler_tick_seconds())

    async def _expire_overdue_tasks(self) -> None:
        now_iso = self._iso(self._now())
        async with self._db_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE supervision_tasks
                    SET status = 'expired', updated_at = ?
                    WHERE status = 'active' AND schedule_kind = 'once' AND end_at != '' AND end_at <= ?
                    """,
                    (now_iso, now_iso),
                )
                conn.commit()
                self._sync_active_tasks_to_settings_sync(conn)

    async def _run_due_broadcasts(self) -> None:
        now = self._now()
        async with self._db_lock:
            with self._connect() as conn:
                rows = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT kind, enabled, session_id, session_label, time_hhmm, content, last_sent_at
                        FROM broadcast_jobs
                        WHERE enabled = 1 AND time_hhmm != '' AND content != '' AND session_id != ''
                        """
                    ).fetchall()
                ]

        for row in rows:
            kind = str(row["kind"])
            if kind == "update" and not self._update_feature_enabled():
                continue
            if kind == "preview" and not self._preview_feature_enabled():
                continue

            scheduled_time = self._normalize_hhmm(str(row["time_hhmm"] or ""))
            if not scheduled_time:
                continue
            hour = int(scheduled_time.split(":")[0])
            minute = int(scheduled_time.split(":")[1])
            due_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now < due_at:
                continue

            last_sent_at = self._parse_dt(str(row["last_sent_at"] or ""))
            if last_sent_at and last_sent_at.date() == now.date():
                continue

            message = self._render_broadcast_text(kind, str(row["content"] or ""))
            try:
                await self.context.send_message(
                    str(row["session_id"]),
                    MessageChain().message(message),
                )
            except Exception as exc:
                logger.warning(
                    f"WorkSupervisor failed to send scheduled {kind} broadcast: {exc}",
                    exc_info=True,
                )
                continue

            async with self._db_lock:
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE broadcast_jobs SET last_sent_at = ?, updated_at = ? WHERE kind = ?",
                        (self._iso(now), self._iso(now), kind),
                    )
                    conn.commit()

    def _is_settings_seeded_task(self, task: dict[str, Any]) -> bool:
        return self._normalize_settings_task_key(task.get("settings_task_key")).startswith("cfg-")

    def _settings_task_due_initial_push(
        self,
        conn: sqlite3.Connection,
        task: dict[str, Any],
        now: datetime,
    ) -> bool:
        if str(task.get("status") or "") != "active":
            return False
        if not self._is_settings_seeded_task(task):
            return False
        if not str(task.get("trigger_session_id") or "").strip():
            return False
        start_at = self._parse_dt(task.get("start_at"))
        if start_at and start_at > now:
            return False
        if self._task_reminder_count_exhausted(conn, task, now):
            return False

        last_reminded_at = self._parse_dt(task.get("last_reminded_at"))
        if last_reminded_at is None:
            return True

        if self._normalize_schedule_kind(task.get("schedule_kind")) != "daily":
            return False
        cycle_start = self._task_cycle_start(task, now)
        return bool(cycle_start and last_reminded_at < cycle_start)

    def _build_reminder_message_chain(
        self,
        task: dict[str, Any],
        reminder_text: str,
    ) -> MessageChain:
        message_chain = MessageChain()
        session_id = str(task.get("trigger_session_id") or "").strip()
        if ":GroupMessage:" in session_id or str(task.get("trigger_group_id") or "").strip():
            message_chain.at(
                name=str(task.get("target_user_name") or task.get("target_user_id") or "用户"),
                qq=str(task.get("target_user_id") or ""),
            ).message(" ")
        message_chain.message(reminder_text)
        return message_chain

    async def _run_due_settings_initial_reminders(self) -> None:
        now = self._now()
        async with self._db_lock:
            with self._connect() as conn:
                tasks = [
                    task
                    for task in self._list_active_tasks(conn)
                    if self._settings_task_due_initial_push(conn, task, now)
                ]

        for task in tasks:
            selected_todos = self._pick_todos(task)
            session_id = str(task.get("trigger_session_id") or "").strip()
            if not session_id:
                continue
            reminder_text = await self._build_reminder_text_for_session(
                task,
                selected_todos,
                session_id,
            )
            try:
                await self.context.send_message(
                    session_id,
                    self._build_reminder_message_chain(task, reminder_text),
                )
            except Exception as exc:
                logger.warning(
                    f"WorkSupervisor failed to send settings initial reminder: {exc}",
                    exc_info=True,
                )
                continue

            async with self._db_lock:
                with self._connect() as conn:
                    self._touch_reminder_record(
                        conn,
                        int(task["id"]),
                        reminder_text=reminder_text,
                        target_user_id=str(task.get("target_user_id") or ""),
                        trigger_session_id=session_id,
                        now=now,
                    )

    # ---- 指令解析 ----
    def _clean_payload_after_mentions(self, payload: str) -> str:
        text = str(payload or "").strip()
        if not text:
            return ""
        if text.startswith("@"):
            return re.sub(r"^@\S+\s*", "", text, count=1).strip()
        return text

    def _parse_duration_seconds(self, text: str, default_seconds: int) -> int:
        raw = str(text or "").strip().lower()
        if not raw:
            return default_seconds
        raw = (
            raw.replace("小时", "h")
            .replace("时", "h")
            .replace("分钟", "m")
            .replace("分", "m")
            .replace("天", "d")
            .replace("秒", "s")
        )
        if raw.isdigit():
            return max(int(raw) * 60, 60)

        total = 0
        for value, unit in re.findall(r"(\d+)\s*(d|h|m|s)", raw):
            number = int(value)
            if unit == "d":
                total += number * 86400
            elif unit == "h":
                total += number * 3600
            elif unit == "m":
                total += number * 60
            elif unit == "s":
                total += number
        if total > 0:
            return total
        return default_seconds

    def _parse_todo_items(self, text: str) -> list[str]:
        items = self._parse_text_list(
            str(text or "").replace("、", "\n").replace("|", "\n")
        )
        deduped: list[str] = []
        for item in items:
            if item and item not in deduped:
                deduped.append(item)
        return deduped[: self._max_todo_items_per_task()]

    def _extract_key_value_fields(
        self,
        text: str,
        aliases: dict[str, str],
    ) -> tuple[str, dict[str, str]]:
        # Command UX uses Chinese key-value segments. Values are captured until the
        # next known key, so task names and content can still contain spaces.
        if not text:
            return "", {}

        alias_lookup = {str(key).lower(): value for key, value in aliases.items()}
        key_pattern = "|".join(
            re.escape(key) for key in sorted(alias_lookup, key=len, reverse=True)
        )
        pattern = re.compile(rf"(?P<key>{key_pattern})\s*(?:=|:|：)\s*", re.I)
        matches = list(pattern.finditer(text))
        if not matches:
            return text.strip(), {}

        prefix = text[: matches[0].start()].strip(" |，,；;")
        fields: dict[str, str] = {}
        for index, match in enumerate(matches):
            raw_key = str(match.group("key") or "").lower()
            canonical_key = alias_lookup.get(raw_key)
            if canonical_key is None:
                continue
            value_start = match.end()
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            value = text[value_start:value_end].strip(" |，,；;")
            if value:
                fields[canonical_key] = value
        return prefix, fields

    def _parse_start_payload_legacy_pipe(self, text: str) -> dict[str, Any] | None:
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if not parts:
            return None

        title = parts[0]
        todos: list[str] = []
        duration_seconds = self._default_duration_minutes() * 60
        cooldown_seconds = self._default_cooldown_minutes() * 60
        todo_pick_count = self._default_todo_pick_count()

        for part in parts[1:]:
            lowered = part.lower()
            if lowered.startswith(("时长=", "持续=", "duration=", "到期=")):
                duration_seconds = self._parse_duration_seconds(
                    part.split("=", 1)[1],
                    duration_seconds,
                )
            elif lowered.startswith(("冷却=", "cooldown=")):
                cooldown_seconds = self._parse_duration_seconds(
                    part.split("=", 1)[1],
                    cooldown_seconds,
                )
            elif lowered.startswith(("抽取=", "展示=", "pick=")):
                try:
                    todo_pick_count = max(int(part.split("=", 1)[1]), 1)
                except (TypeError, ValueError):
                    todo_pick_count = self._default_todo_pick_count()
            elif lowered.startswith(("待办=", "事项=", "todo=")):
                todos = self._parse_todo_items(part.split("=", 1)[1])
            elif not todos:
                todos = self._parse_todo_items(part)

        return self._build_start_payload(
            title=title,
            todos=todos,
            duration_seconds=duration_seconds,
            cooldown_seconds=cooldown_seconds,
            todo_pick_count=todo_pick_count,
        )

    def _build_start_payload(
        self,
        *,
        title: str,
        todos: list[str],
        duration_seconds: int,
        cooldown_seconds: int,
        todo_pick_count: int,
    ) -> dict[str, Any] | None:
        normalized_title = self._normalize_command_text(title).strip(" |，,；;")
        if not normalized_title:
            return None

        todo_pick_count = min(todo_pick_count, max(len(todos), 1), 5)
        return {
            "title": normalized_title,
            "todos": todos,
            "duration_seconds": max(duration_seconds, 60),
            "cooldown_seconds": max(cooldown_seconds, 0),
            "todo_pick_count": max(todo_pick_count, 1),
        }

    def _parse_start_payload(self, payload: str) -> dict[str, Any] | None:
        text = str(payload or "").strip().replace("｜", "|")
        if not text:
            return None

        if "|" in text:
            return self._parse_start_payload_legacy_pipe(text)

        prefix, fields = self._extract_key_value_fields(
            text,
            {
                "任务": "title",
                "标题": "title",
                "task": "title",
                "title": "title",
                "待办": "todos",
                "事项": "todos",
                "todo": "todos",
                "todos": "todos",
                "时长": "duration",
                "持续": "duration",
                "duration": "duration",
                "到期": "duration",
                "冷却": "cooldown",
                "cooldown": "cooldown",
                "抽取": "pick",
                "展示": "pick",
                "pick": "pick",
            },
        )

        title = fields.get("title") or prefix
        todos = self._parse_todo_items(fields.get("todos", ""))
        duration_seconds = self._default_duration_minutes() * 60
        cooldown_seconds = self._default_cooldown_minutes() * 60
        todo_pick_count = self._default_todo_pick_count()

        if "duration" in fields:
            duration_seconds = self._parse_duration_seconds(fields["duration"], duration_seconds)
        if "cooldown" in fields:
            cooldown_seconds = self._parse_duration_seconds(fields["cooldown"], cooldown_seconds)
        if "pick" in fields:
            try:
                todo_pick_count = max(int(fields["pick"]), 1)
            except (TypeError, ValueError):
                todo_pick_count = self._default_todo_pick_count()

        return self._build_start_payload(
            title=title,
            todos=todos,
            duration_seconds=duration_seconds,
            cooldown_seconds=cooldown_seconds,
            todo_pick_count=todo_pick_count,
        )

    def _parse_broadcast_setting(self, payload: str) -> tuple[str, str] | None:
        text = str(payload or "").strip()
        _, fields = self._extract_key_value_fields(
            text,
            {
                "时间": "time",
                "发送时间": "time",
                "time": "time",
                "内容": "content",
                "文案": "content",
                "content": "content",
            },
        )
        if fields:
            hhmm = self._normalize_hhmm(fields.get("time", ""))
            content = str(fields.get("content", "")).strip()
            if hhmm and content:
                return hhmm, content
            return None

        match = re.match(r"^(\d{1,2}:\d{2})\s+(.+)$", text, re.S)
        if not match:
            return None
        hhmm = self._normalize_hhmm(match.group(1))
        content = match.group(2).strip()
        if not hhmm or not content:
            return None
        return hhmm, content

    # ---- 设置页任务同步 ----
    def _normalize_settings_task_key(self, value: Any) -> str:
        raw = str(value or "").strip()
        normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "-", raw).strip("-")
        return normalized[:80]

    def _new_settings_task_key(self, prefix: str, *parts: Any) -> str:
        seed = "|".join(str(part or "") for part in parts)
        digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"{prefix}-{digest}"

    def _derive_settings_session(
        self,
        item: dict[str, Any],
        target_user_id: str,
    ) -> tuple[str, str]:
        session_id = str(item.get("session_id") or "").strip()
        group_id = str(item.get("group_id") or "").strip()
        if session_id:
            match = re.search(r":GroupMessage:([^:]+)$", session_id)
            if match and not group_id:
                group_id = match.group(1)
            return session_id, group_id

        platform_id = str(item.get("platform_id") or "napcat").strip() or "napcat"
        session_type = str(item.get("session_type") or "").strip()
        if group_id:
            return f"{platform_id}:GroupMessage:{group_id}", group_id
        if session_type.lower() == "groupmessage":
            return "", group_id
        return f"{platform_id}:FriendMessage:{target_user_id}", ""

    def _target_from_settings_item(self, item: dict[str, Any]) -> str:
        target_user_id = str(item.get("target_user_id") or "").strip()
        if target_user_id:
            return target_user_id
        session_id = str(item.get("session_id") or "").strip()
        match = re.search(r":FriendMessage:([^:]+)$", session_id)
        return match.group(1).strip() if match else ""

    def _parse_settings_task_item(
        self,
        item: dict[str, Any],
        index: int,
        now: datetime,
    ) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
        if not self._get_bool_from_value(item.get("enabled", True), True):
            return None

        target_user_id = self._target_from_settings_item(item)
        task_title = self._normalize_command_text(item.get("task_title", ""))
        if not target_user_id or not task_title:
            logger.warning(
                "WorkSupervisor settings task ignored: target_user_id and task_title are required."
            )
            return None

        session_id, group_id = self._derive_settings_session(item, target_user_id)
        if not session_id:
            logger.warning(
                "WorkSupervisor settings task ignored: session_id is required for group tasks."
            )
            return None

        raw_key = self._normalize_settings_task_key(item.get("task_key"))
        task_key = raw_key or self._new_settings_task_key(
            "cfg",
            index,
            target_user_id,
            session_id,
            task_title,
            item.get("end_at") or item.get("deadline_at") or item.get("duration") or "",
        )
        schedule_kind, duration_seconds, duration_label = self._parse_settings_duration_spec(
            item.get("duration")
        )
        cooldown_seconds = self._parse_duration_seconds(
            str(item.get("reminder_interval") or item.get("cooldown") or ""),
            self._default_cooldown_minutes() * 60,
        )
        try:
            reminder_limit = max(int(item.get("reminder_count") or 0), 0)
        except (TypeError, ValueError):
            reminder_limit = 0
        todo_pick_count: int | None = None
        if item.get("todo_pick_count") not in {None, ""}:
            try:
                todo_pick_count = max(int(item.get("todo_pick_count")), 1)
            except (TypeError, ValueError):
                todo_pick_count = self._default_todo_pick_count()
        todo_items = self._parse_todo_items(item.get("todo_items", ""))
        start_at = self._parse_settings_datetime(item.get("start_at")) or now
        end_at = self._parse_settings_datetime(item.get("end_at") or item.get("deadline_at"))
        if end_at is not None:
            schedule_kind = "once"
            duration_seconds = max(int((end_at - start_at).total_seconds()), 60)
            duration_label = self._format_compact_duration_seconds(duration_seconds)
        elif schedule_kind == "once" and duration_seconds is not None:
            end_at = start_at + timedelta(seconds=duration_seconds)
            duration_label = self._format_compact_duration_seconds(duration_seconds)

        if end_at is not None and end_at <= start_at:
            logger.warning(
                "WorkSupervisor settings task ignored: end_at must be later than start_at."
            )
            return None
        if end_at is None and schedule_kind == "once":
            logger.warning(
                "WorkSupervisor settings task ignored: duration or end_at is required."
            )
            return None

        parsed = {
            "target_user_id": target_user_id,
            "target_user_name": str(item.get("target_user_name") or target_user_id).strip() or target_user_id,
            "trigger_session_id": session_id,
            "trigger_group_id": group_id,
            "created_by_user_id": str(item.get("created_by_user_id") or "bot").strip() or "bot",
            "created_by_user_name": str(item.get("created_by_user_name") or "机器人").strip() or "机器人",
            "task_title": task_title,
            "todo_items": todo_items,
            "schedule_kind": schedule_kind,
            "start_at": start_at,
            "duration_seconds": max(int(duration_seconds or 0), 0),
            "duration_label": duration_label,
            "cooldown_seconds": max(cooldown_seconds, 0),
            "reminder_limit": reminder_limit,
            "todo_pick_count": None
            if todo_pick_count is None
            else min(max(todo_pick_count, 1), max(len(todo_items), 1), 5),
            "end_at": end_at,
        }
        normalized_item = {
            "__template_key": "supervision_task",
            "task_key": task_key,
            "enabled": True,
            "platform_id": str(item.get("platform_id") or "napcat").strip() or "napcat",
            "session_type": str(item.get("session_type") or ("GroupMessage" if group_id else "FriendMessage")).strip(),
            "session_id": session_id,
            "group_id": group_id,
            "target_user_id": target_user_id,
            "target_user_name": parsed["target_user_name"],
            "created_by_user_id": parsed["created_by_user_id"],
            "created_by_user_name": parsed["created_by_user_name"],
            "task_title": task_title,
            "todo_items": "\n".join(todo_items),
            "start_at": self._format_time(start_at),
            "duration": parsed["duration_label"] or "",
            "end_at": self._format_time(end_at) if end_at else "",
            "reminder_interval": self._format_compact_duration_seconds(parsed["cooldown_seconds"]),
            "reminder_count": parsed["reminder_limit"],
        }
        if parsed["todo_pick_count"] is not None:
            normalized_item["todo_pick_count"] = parsed["todo_pick_count"]
        return task_key, parsed, normalized_item

    def _task_to_settings_item(
        self,
        conn: sqlite3.Connection,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = int(task["id"])
        task_key = self._normalize_settings_task_key(task.get("settings_task_key"))
        if not task_key:
            task_key = f"cmd-{task_id}"
            conn.execute(
                "UPDATE supervision_tasks SET settings_task_key = ?, updated_at = ? WHERE id = ?",
                (task_key, self._iso(self._now()), task_id),
            )
            conn.commit()

        try:
            todos = json.loads(str(task.get("todo_items_json") or "[]"))
        except Exception:
            todos = []
        todo_items = [str(item).strip() for item in todos if str(item).strip()] if isinstance(todos, list) else []
        start_at = self._parse_dt(task.get("start_at")) or self._now()
        end_at = self._parse_dt(task.get("end_at"))
        schedule_kind = self._normalize_schedule_kind(task.get("schedule_kind"))
        duration_label = self._format_schedule_duration_label(task)
        session_id = str(task.get("trigger_session_id") or "")
        group_id = str(task.get("trigger_group_id") or "")
        if not group_id:
            match = re.search(r":GroupMessage:([^:]+)$", session_id)
            if match:
                group_id = match.group(1)

        platform_id = session_id.split(":", 1)[0] if ":" in session_id else "napcat"
        session_type = "GroupMessage" if ":GroupMessage:" in session_id else "FriendMessage"
        return {
            "__template_key": "supervision_task",
            "task_key": task_key,
            "enabled": True,
            "platform_id": platform_id,
            "session_type": session_type,
            "session_id": session_id,
            "group_id": group_id,
            "target_user_id": str(task.get("target_user_id") or ""),
            "target_user_name": str(task.get("target_user_name") or task.get("target_user_id") or ""),
            "created_by_user_id": str(task.get("created_by_user_id") or ""),
            "created_by_user_name": str(task.get("created_by_user_name") or ""),
            "task_title": str(task.get("task_title") or ""),
            "todo_items": "\n".join(todo_items),
            "start_at": self._format_time(start_at),
            "duration": duration_label,
            "end_at": self._format_time(end_at) if schedule_kind == "once" and end_at else "",
            "reminder_interval": self._format_compact_duration_seconds(int(task.get("cooldown_seconds") or 0)),
            "reminder_count": int(task.get("reminder_limit") or 0),
            "todo_pick_count": int(task.get("todo_pick_count") or 1),
        }

    def _sync_active_tasks_to_settings_sync(self, conn: sqlite3.Connection) -> None:
        items = [self._task_to_settings_item(conn, task) for task in self._list_active_tasks(conn)]
        if items == self._settings_tasks_config():
            self._last_settings_tasks_signature = self._settings_tasks_signature()
            return
        self._save_plugin_config({"settings_tasks": items})
        self._last_settings_tasks_signature = self._settings_tasks_signature()

    def _sync_settings_tasks_from_config_sync(self, force: bool = False) -> None:
        signature = self._settings_tasks_signature()
        if not force and signature == self._last_settings_tasks_signature:
            return

        now = self._now()
        settings_items = self._settings_tasks_config()
        desired_keys: set[str] = set()
        with self._connect() as conn:
            for index, item in enumerate(settings_items):
                parsed_item = self._parse_settings_task_item(item, index, now)
                if parsed_item is None:
                    continue
                task_key, parsed, _normalized_item = parsed_item
                desired_keys.add(task_key)

                active_by_key = self._find_active_task_by_settings_key(conn, task_key)
                if (
                    parsed["schedule_kind"] == "once"
                    and parsed["end_at"] is not None
                    and parsed["end_at"] <= now
                ):
                    if active_by_key:
                        self._mark_task_status(conn, int(active_by_key["id"]), "expired", now)
                    continue

                active_by_target = self._find_active_task_by_target(conn, parsed["target_user_id"])
                if active_by_target and str(active_by_target.get("settings_task_key") or "") not in {"", task_key}:
                    logger.warning(
                        "WorkSupervisor settings task ignored: target %s already has an active task.",
                        parsed["target_user_id"],
                    )
                    continue

                if active_by_key:
                    todo_pick_count = (
                        parsed["todo_pick_count"]
                        if parsed["todo_pick_count"] is not None
                        else int(active_by_key.get("todo_pick_count") or self._default_todo_pick_count())
                    )
                    self._update_active_task_from_settings(
                        conn,
                        int(active_by_key["id"]),
                        target_user_id=parsed["target_user_id"],
                        target_user_name=parsed["target_user_name"],
                        trigger_session_id=parsed["trigger_session_id"],
                        trigger_group_id=parsed["trigger_group_id"],
                        created_by_user_id=parsed["created_by_user_id"],
                        created_by_user_name=parsed["created_by_user_name"],
                        task_title=parsed["task_title"],
                        todo_items=parsed["todo_items"],
                        schedule_kind=parsed["schedule_kind"],
                        start_at=parsed["start_at"],
                        end_at=parsed["end_at"],
                        cooldown_seconds=parsed["cooldown_seconds"],
                        reminder_limit=parsed["reminder_limit"],
                        todo_pick_count=todo_pick_count,
                        now=now,
                    )
                    continue

                if active_by_target and not str(active_by_target.get("settings_task_key") or ""):
                    conn.execute(
                        "UPDATE supervision_tasks SET settings_task_key = ?, updated_at = ? WHERE id = ?",
                        (task_key, self._iso(now), int(active_by_target["id"])),
                    )
                    conn.commit()
                    active_by_key = self._find_active_task_by_settings_key(conn, task_key)
                    if active_by_key:
                        todo_pick_count = (
                            parsed["todo_pick_count"]
                            if parsed["todo_pick_count"] is not None
                            else int(active_by_key.get("todo_pick_count") or self._default_todo_pick_count())
                        )
                        self._update_active_task_from_settings(
                            conn,
                            int(active_by_key["id"]),
                            target_user_id=parsed["target_user_id"],
                            target_user_name=parsed["target_user_name"],
                            trigger_session_id=parsed["trigger_session_id"],
                            trigger_group_id=parsed["trigger_group_id"],
                            created_by_user_id=parsed["created_by_user_id"],
                            created_by_user_name=parsed["created_by_user_name"],
                            task_title=parsed["task_title"],
                            todo_items=parsed["todo_items"],
                            schedule_kind=parsed["schedule_kind"],
                            start_at=parsed["start_at"],
                            end_at=parsed["end_at"],
                            cooldown_seconds=parsed["cooldown_seconds"],
                            reminder_limit=parsed["reminder_limit"],
                            todo_pick_count=todo_pick_count,
                            now=now,
                        )
                    continue

                todo_pick_count = (
                    parsed["todo_pick_count"]
                    if parsed["todo_pick_count"] is not None
                    else self._default_todo_pick_count()
                )
                self._save_task(
                    conn,
                    target_user_id=parsed["target_user_id"],
                    target_user_name=parsed["target_user_name"],
                    trigger_session_id=parsed["trigger_session_id"],
                    trigger_group_id=parsed["trigger_group_id"],
                    created_by_user_id=parsed["created_by_user_id"],
                    created_by_user_name=parsed["created_by_user_name"],
                    task_title=parsed["task_title"],
                    todo_items=parsed["todo_items"],
                    schedule_kind=parsed["schedule_kind"],
                    duration_seconds=parsed["duration_seconds"],
                    cooldown_seconds=parsed["cooldown_seconds"],
                    reminder_limit=parsed["reminder_limit"],
                    todo_pick_count=todo_pick_count,
                    now=now,
                    start_at=parsed["start_at"],
                    end_at=parsed["end_at"],
                    settings_task_key=task_key,
                )

            for task in self._list_active_tasks(conn):
                task_key = self._normalize_settings_task_key(task.get("settings_task_key"))
                if task_key and task_key not in desired_keys:
                    self._mark_task_status(conn, int(task["id"]), "cancelled", now)

            self._sync_active_tasks_to_settings_sync(conn)

    async def _sync_settings_tasks_from_config(self, force: bool = False) -> None:
        async with self._db_lock:
            self._sync_settings_tasks_from_config_sync(force=force)

    async def _sync_active_tasks_to_settings(self) -> None:
        async with self._db_lock:
            with self._connect() as conn:
                self._sync_active_tasks_to_settings_sync(conn)

    def _broadcast_template_kind(self, template_key: Any) -> str:
        normalized = self._normalize_command_text(template_key)
        if normalized in {"update_broadcast", "update"}:
            return "update"
        if normalized in {"preview_broadcast", "preview"}:
            return "preview"
        return ""

    def _parse_broadcast_settings_item(
        self,
        item: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
        kind = self._broadcast_template_kind(item.get("__template_key")) or self._normalize_command_text(
            item.get("kind")
        )
        if kind not in {"update", "preview"}:
            return None

        time_hhmm = self._normalize_hhmm(str(item.get("time_hhmm") or item.get("time") or ""))
        session_id = str(item.get("session_id") or "").strip()
        session_label = str(item.get("session_label") or session_id).strip() or session_id
        content = str(item.get("content") or "").strip()
        enabled = self._get_bool_from_value(item.get("enabled"), True)
        normalized_item = {
            "__template_key": f"{kind}_broadcast",
            "enabled": enabled,
            "session_id": session_id,
            "session_label": session_label,
            "time_hhmm": time_hhmm,
            "content": content,
        }
        parsed = {
            "enabled": enabled,
            "session_id": session_id,
            "session_label": session_label,
            "time_hhmm": time_hhmm,
            "content": content,
        }
        return kind, parsed, normalized_item

    def _broadcast_job_to_settings_item(self, row: dict[str, Any]) -> dict[str, Any]:
        kind = str(row.get("kind") or "").strip() or "update"
        session_id = str(row.get("session_id") or "").strip()
        session_label = str(row.get("session_label") or session_id).strip() or session_id
        return {
            "__template_key": f"{kind}_broadcast",
            "enabled": bool(int(row.get("enabled") or 0)),
            "session_id": session_id,
            "session_label": session_label,
            "time_hhmm": self._normalize_hhmm(str(row.get("time_hhmm") or "")),
            "content": str(row.get("content") or "").strip(),
        }

    def _list_broadcast_jobs(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM broadcast_jobs
                ORDER BY kind
                """
            ).fetchall()
        ]

    def _sync_broadcast_jobs_to_config_sync(self, conn: sqlite3.Connection) -> None:
        rows = {str(row.get("kind") or ""): row for row in self._list_broadcast_jobs(conn)}
        items: list[dict[str, Any]] = []
        for kind in ("update", "preview"):
            row = rows.get(kind)
            if row is None:
                continue
            if not any(
                str(row.get(field) or "").strip()
                for field in ("session_id", "session_label", "time_hhmm", "content")
            ):
                continue
            items.append(self._broadcast_job_to_settings_item(row))

        if items == self._broadcast_settings_config():
            self._last_broadcast_settings_signature = self._broadcast_settings_signature()
            return
        self._save_plugin_config({"broadcast_settings": items})
        self._last_broadcast_settings_signature = self._broadcast_settings_signature()

    def _delete_broadcast_job(self, conn: sqlite3.Connection, kind: str) -> None:
        conn.execute("DELETE FROM broadcast_jobs WHERE kind = ?", (kind,))
        conn.commit()

    def _bootstrap_broadcast_jobs_sync(self) -> None:
        if self._broadcast_settings_config():
            self._sync_broadcast_jobs_from_config_sync(force=True)
            return
        with self._connect() as conn:
            self._sync_broadcast_jobs_to_config_sync(conn)
        self._last_broadcast_settings_signature = self._broadcast_settings_signature()

    def _sync_broadcast_jobs_from_config_sync(self, force: bool = False) -> None:
        signature = self._broadcast_settings_signature()
        if not force and signature == self._last_broadcast_settings_signature:
            return

        parsed_by_kind: dict[str, dict[str, Any]] = {}
        for item in self._broadcast_settings_config():
            parsed_item = self._parse_broadcast_settings_item(item)
            if parsed_item is None:
                continue
            kind, parsed, _normalized_item = parsed_item
            parsed_by_kind[kind] = parsed

        now = self._now()
        with self._connect() as conn:
            existing = {str(row.get("kind") or ""): row for row in self._list_broadcast_jobs(conn)}
            for kind in ("update", "preview"):
                parsed = parsed_by_kind.get(kind)
                if parsed is None:
                    if existing.get(kind) is not None:
                        self._delete_broadcast_job(conn, kind)
                    continue

                if not any(
                    str(parsed.get(field) or "").strip()
                    for field in ("session_id", "time_hhmm", "content")
                ):
                    if existing.get(kind) is not None:
                        self._delete_broadcast_job(conn, kind)
                    continue

                existing_row = existing.get(kind) or {}
                self._upsert_broadcast_job(
                    conn,
                    kind=kind,
                    enabled=bool(parsed["enabled"]),
                    session_id=str(parsed["session_id"]),
                    session_label=str(parsed["session_label"]),
                    time_hhmm=str(parsed["time_hhmm"]),
                    content=str(parsed["content"]),
                    updated_by_user_id=str(existing_row.get("updated_by_user_id") or "config"),
                    updated_by_user_name=str(existing_row.get("updated_by_user_name") or "设置页"),
                    now=now,
                )

            self._sync_broadcast_jobs_to_config_sync(conn)

    async def _sync_broadcast_jobs_from_config(self, force: bool = False) -> None:
        async with self._db_lock:
            self._sync_broadcast_jobs_from_config_sync(force=force)

    # ---- 数据访问 ----
    def _find_active_task_by_target(
        self,
        conn: sqlite3.Connection,
        target_user_id: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT *
            FROM supervision_tasks
            WHERE target_user_id = ? AND status = 'active'
            ORDER BY id DESC
            LIMIT 1
            """,
            (target_user_id,),
        ).fetchone()
        return dict(row) if row else None

    def _find_active_task_for_session(
        self,
        conn: sqlite3.Connection,
        target_user_id: str,
        trigger_session_id: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT *
            FROM supervision_tasks
            WHERE target_user_id = ? AND status = 'active' AND trigger_session_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (target_user_id, trigger_session_id),
        ).fetchone()
        return dict(row) if row else None

    def _task_matches_event_scope(
        self,
        task: dict[str, Any],
        *,
        session_id: str,
        group_id: str,
    ) -> bool:
        task_session_id = str(task.get("trigger_session_id") or "").strip()
        if task_session_id and task_session_id == session_id:
            return True

        task_group_id = str(task.get("trigger_group_id") or "").strip()
        if task_group_id:
            return bool(group_id) and task_group_id == group_id

        # Private/self supervision should follow the target user across chats.
        # Group supervision stays pinned to the originating group.
        return True

    def _find_active_task_for_event(
        self,
        conn: sqlite3.Connection,
        target_user_id: str,
        *,
        session_id: str,
        group_id: str,
    ) -> dict[str, Any] | None:
        rows = conn.execute(
            """
            SELECT *
            FROM supervision_tasks
            WHERE target_user_id = ? AND status = 'active'
            ORDER BY id DESC
            """,
            (target_user_id,),
        ).fetchall()
        for row in rows:
            task = dict(row)
            if self._task_matches_event_scope(
                task,
                session_id=session_id,
                group_id=group_id,
            ):
                return task
        return None

    def _find_task_by_id(
        self,
        conn: sqlite3.Connection,
        task_id: int,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM supervision_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        return dict(row) if row else None

    def _count_task_reminders(
        self,
        conn: sqlite3.Connection,
        task_id: int,
        *,
        since: datetime | None = None,
    ) -> int:
        if since is None:
            row = conn.execute(
                "SELECT COUNT(1) AS total FROM reminder_logs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT COUNT(1) AS total
                FROM reminder_logs
                WHERE task_id = ? AND created_at >= ?
                """,
                (task_id, self._iso(since)),
            ).fetchone()
        return int(row["total"] or 0) if row else 0

    def _find_active_task_by_settings_key(
        self,
        conn: sqlite3.Connection,
        settings_task_key: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT *
            FROM supervision_tasks
            WHERE settings_task_key = ? AND status = 'active'
            ORDER BY id DESC
            LIMIT 1
            """,
            (settings_task_key,),
        ).fetchone()
        return dict(row) if row else None

    def _list_active_tasks(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM supervision_tasks
                WHERE status = 'active'
                ORDER BY id ASC
                """
            )
        ]

    def _save_task(
        self,
        conn: sqlite3.Connection,
        *,
        target_user_id: str,
        target_user_name: str,
        trigger_session_id: str,
        trigger_group_id: str,
        created_by_user_id: str,
        created_by_user_name: str,
        task_title: str,
        todo_items: list[str],
        duration_seconds: int,
        cooldown_seconds: int,
        todo_pick_count: int,
        now: datetime,
        schedule_kind: str = "once",
        reminder_limit: int = 0,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        settings_task_key: str = "",
    ) -> int:
        start_at = start_at or now
        normalized_schedule_kind = self._normalize_schedule_kind(schedule_kind)
        resolved_end_at = end_at
        if resolved_end_at is None and normalized_schedule_kind == "once":
            resolved_end_at = start_at + timedelta(seconds=duration_seconds)
        cursor = conn.execute(
            """
            INSERT INTO supervision_tasks (
                target_user_id, target_user_name, trigger_session_id, trigger_group_id,
                created_by_user_id, created_by_user_name, task_title, todo_items_json,
                status, start_at, end_at, schedule_kind, cooldown_seconds, reminder_limit, todo_pick_count,
                last_reminded_at, completed_at, cancelled_at, created_at, updated_at,
                settings_task_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, '', '', '', ?, ?, ?)
            """,
            (
                target_user_id,
                target_user_name,
                trigger_session_id,
                trigger_group_id,
                created_by_user_id,
                created_by_user_name,
                task_title,
                json.dumps(todo_items, ensure_ascii=False),
                self._iso(start_at),
                self._iso(resolved_end_at) if resolved_end_at else "",
                normalized_schedule_kind,
                cooldown_seconds,
                max(int(reminder_limit), 0),
                todo_pick_count,
                self._iso(now),
                self._iso(now),
                settings_task_key,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)

    def _update_active_task_from_settings(
        self,
        conn: sqlite3.Connection,
        task_id: int,
        *,
        target_user_id: str,
        target_user_name: str,
        trigger_session_id: str,
        trigger_group_id: str,
        created_by_user_id: str,
        created_by_user_name: str,
        task_title: str,
        todo_items: list[str],
        schedule_kind: str,
        start_at: datetime,
        end_at: datetime | None,
        cooldown_seconds: int,
        reminder_limit: int,
        todo_pick_count: int,
        now: datetime,
    ) -> None:
        conn.execute(
            """
            UPDATE supervision_tasks
            SET target_user_id = ?, target_user_name = ?, trigger_session_id = ?,
                trigger_group_id = ?, created_by_user_id = ?, created_by_user_name = ?,
                task_title = ?, todo_items_json = ?, start_at = ?, end_at = ?, schedule_kind = ?,
                cooldown_seconds = ?, reminder_limit = ?, todo_pick_count = ?, updated_at = ?
            WHERE id = ? AND status = 'active'
            """,
            (
                target_user_id,
                target_user_name,
                trigger_session_id,
                trigger_group_id,
                created_by_user_id,
                created_by_user_name,
                task_title,
                json.dumps(todo_items, ensure_ascii=False),
                self._iso(start_at),
                self._iso(end_at) if end_at else "",
                self._normalize_schedule_kind(schedule_kind),
                cooldown_seconds,
                max(int(reminder_limit), 0),
                todo_pick_count,
                self._iso(now),
                task_id,
            ),
        )
        conn.commit()

    def _mark_task_status(
        self,
        conn: sqlite3.Connection,
        task_id: int,
        status: str,
        now: datetime,
    ) -> None:
        completed_at = self._iso(now) if status == "completed" else ""
        cancelled_at = self._iso(now) if status == "cancelled" else ""
        conn.execute(
            """
            UPDATE supervision_tasks
            SET status = ?, completed_at = ?, cancelled_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, completed_at, cancelled_at, self._iso(now), task_id),
        )
        conn.commit()

    def _touch_reminder_record(
        self,
        conn: sqlite3.Connection,
        task_id: int,
        *,
        reminder_text: str,
        target_user_id: str,
        trigger_session_id: str,
        now: datetime,
    ) -> None:
        now_iso = self._iso(now)
        conn.execute(
            "UPDATE supervision_tasks SET last_reminded_at = ?, updated_at = ? WHERE id = ?",
            (now_iso, now_iso, task_id),
        )
        conn.execute(
            """
            INSERT INTO reminder_logs(task_id, target_user_id, trigger_session_id, reminder_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                task_id,
                target_user_id,
                trigger_session_id,
                reminder_text,
                now_iso,
            ),
        )
        conn.commit()

    def _touch_reminder(
        self,
        conn: sqlite3.Connection,
        task_id: int,
        reminder_text: str,
        event: AstrMessageEvent,
        now: datetime,
    ) -> None:
        self._touch_reminder_record(
            conn,
            task_id,
            reminder_text=reminder_text,
            target_user_id=str(event.get_sender_id() or "").strip(),
            trigger_session_id=str(event.unified_msg_origin or "").strip(),
            now=now,
        )

    def _upsert_broadcast_job(
        self,
        conn: sqlite3.Connection,
        *,
        kind: str,
        enabled: bool,
        session_id: str,
        session_label: str,
        time_hhmm: str,
        content: str,
        updated_by_user_id: str,
        updated_by_user_name: str,
        now: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO broadcast_jobs(
                kind, enabled, session_id, session_label, time_hhmm, content,
                last_sent_at, updated_by_user_id, updated_by_user_name, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?)
            ON CONFLICT(kind) DO UPDATE SET
                enabled = excluded.enabled,
                session_id = excluded.session_id,
                session_label = excluded.session_label,
                time_hhmm = excluded.time_hhmm,
                content = excluded.content,
                updated_by_user_id = excluded.updated_by_user_id,
                updated_by_user_name = excluded.updated_by_user_name,
                updated_at = excluded.updated_at
            """,
            (
                kind,
                1 if enabled else 0,
                session_id,
                session_label,
                time_hhmm,
                content,
                updated_by_user_id,
                updated_by_user_name,
                self._iso(now),
            ),
        )
        conn.commit()

    def _get_broadcast_job(
        self,
        conn: sqlite3.Connection,
        kind: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM broadcast_jobs WHERE kind = ?",
            (kind,),
        ).fetchone()
        return dict(row) if row else None

    # ---- 监督任务状态变更 ----
    def _validate_start_target_permission(
        self,
        event: AstrMessageEvent,
        target_user_id: str,
    ) -> str | None:
        sender_id = str(event.get_sender_id() or "").strip()
        if target_user_id == sender_id:
            return None
        if not self._allow_supervise_others():
            return "当前配置不允许给别人发起监督任务。"
        if not self._allow_non_admin_supervise_others() and not self._is_admin(event):
            return "只有管理员可以在群里给别人发起监督。"
        return None

    def _render_start_success_text(
        self,
        target_user_name: str,
        parsed: dict[str, Any],
        task: dict[str, Any],
    ) -> str:
        lines = [
            f"已开始监督：{target_user_name}",
            f"任务：{parsed['title']}",
            f"持续时间：{self._format_duration_seconds(parsed['duration_seconds'])}",
            f"期间提醒间隔：{self._format_duration_seconds(parsed['cooldown_seconds'])}",
            "提醒次数：不限制",
            f"结束时间：{self._format_time(self._parse_dt(task.get('end_at')))}",
        ]
        if parsed["todos"]:
            lines.append("待办：")
            lines.extend(f"{index}. {item}" for index, item in enumerate(parsed["todos"], start=1))
        return "\n".join(lines)

    async def _create_supervision_result(
        self,
        event: AstrMessageEvent,
        *,
        target_user_id: str,
        target_user_name: str,
        parsed: dict[str, Any],
    ) -> str:
        await self._sync_settings_tasks_from_config()
        permission_error = self._validate_start_target_permission(event, target_user_id)
        if permission_error:
            return permission_error

        sender_id = str(event.get_sender_id() or "").strip()
        sender_name = self._extract_sender_name(event)
        now = self._now()
        async with self._db_lock:
            with self._connect() as conn:
                active_task = self._find_active_task_by_target(conn, target_user_id)
                if active_task:
                    return f"{target_user_name} 当前已经有一个进行中的监督任务：{active_task['task_title']}"

                settings_task_key = self._new_settings_task_key(
                    "cmd",
                    target_user_id,
                    event.unified_msg_origin,
                    parsed["title"],
                    self._iso(now),
                )
                task_id = self._save_task(
                    conn,
                    target_user_id=target_user_id,
                    target_user_name=target_user_name,
                    trigger_session_id=event.unified_msg_origin,
                    trigger_group_id=str(event.get_group_id() or ""),
                    created_by_user_id=sender_id,
                    created_by_user_name=sender_name,
                    task_title=parsed["title"],
                    todo_items=parsed["todos"],
                    duration_seconds=parsed["duration_seconds"],
                    cooldown_seconds=parsed["cooldown_seconds"],
                    todo_pick_count=parsed["todo_pick_count"],
                    now=now,
                    settings_task_key=settings_task_key,
                )
                task = self._find_task_by_id(conn, task_id)
                self._sync_active_tasks_to_settings_sync(conn)

        if task is None:
            return "创建监督任务失败，请稍后再试。"
        return self._render_start_success_text(target_user_name, parsed, task)

    async def _status_supervision_result(
        self,
        event: AstrMessageEvent,
        *,
        target_user_id: str,
    ) -> str:
        await self._sync_settings_tasks_from_config()
        await self._ensure_scheduler_started()
        await self._expire_overdue_tasks()

        async with self._db_lock:
            with self._connect() as conn:
                task = self._find_active_task_by_target(conn, target_user_id)
                if task is not None:
                    since = None
                    if self._normalize_schedule_kind(task.get("schedule_kind")) == "daily":
                        since = self._task_cycle_start(task, self._now())
                    task["_reminder_sent"] = self._count_task_reminders(
                        conn,
                        int(task["id"]),
                        since=since,
                    )

        if task is None:
            return "当前没有进行中的监督任务。"
        return self._render_task_status(task, self._now())

    async def _complete_supervision_result(
        self,
        event: AstrMessageEvent,
        *,
        target_user_id: str,
    ) -> str | None:
        await self._sync_settings_tasks_from_config()
        await self._ensure_scheduler_started()
        await self._expire_overdue_tasks()

        sender_id = str(event.get_sender_id() or "").strip()
        async with self._db_lock:
            with self._connect() as conn:
                task = self._find_active_task_by_target(conn, target_user_id)
                if task is None:
                    return "没有找到进行中的监督任务。"
                if sender_id not in {str(task["target_user_id"]), str(task["created_by_user_id"])} and not self._is_admin(event):
                    return "只有目标本人、任务创建者或管理员可以结束这个监督任务。"
                self._mark_task_status(conn, int(task["id"]), "completed", self._now())
                self._sync_active_tasks_to_settings_sync(conn)

        return f"已结束监督：{task['target_user_name']} 的《{task['task_title']}》"

    # ---- LLM 催促 ----
    async def _get_active_persona_prompt(self, umo: str) -> str:
        persona_manager = getattr(self.context, "persona_manager", None)
        conversation_manager = getattr(self.context, "conversation_manager", None)
        if persona_manager is None:
            return ""

        try:
            conversation_persona_id = None
            if conversation_manager is not None:
                cid = await conversation_manager.get_curr_conversation_id(umo)
                if cid:
                    conversation = await conversation_manager.get_conversation(
                        unified_msg_origin=umo,
                        conversation_id=cid,
                        create_if_not_exists=False,
                    )
                    if conversation is not None:
                        conversation_persona_id = conversation.persona_id

            provider_settings = self.context.get_config(umo=umo).get(
                "provider_settings",
                {},
            )
            _, persona, _, _ = await persona_manager.resolve_selected_persona(
                umo=umo,
                conversation_persona_id=conversation_persona_id,
                platform_name=umo.split(":", 1)[0],
                provider_settings=provider_settings,
            )
            if isinstance(persona, dict):
                prompt = str(persona.get("prompt") or "").strip()
                if prompt:
                    return prompt

            default_persona = await persona_manager.get_default_persona_v3(umo=umo)
            if isinstance(default_persona, dict):
                return str(default_persona.get("prompt") or "").strip()
        except Exception:
            return ""
        return ""

    def _llm_scene_prompt(self) -> str:
        return (
            "你正在扮演一个监督机器人。你的任务不是闲聊，而是用明确、有压迫感但不过界的口吻催人去干活。"
            "你必须结合任务标题、剩余时间和待办事项进行催促。"
            "输出 1 到 3 句简短中文，不要解释系统规则，不要输出 JSON，不要使用 Markdown 列表，不要写免责声明。"
            "允许有性格、有语气，但不要辱骂、不涉及人身攻击、不输出违法违规内容。"
        )

    async def _build_llm_system_prompt(self, umo: str) -> str:
        parts: list[str] = []
        if self._llm_follow_active_persona():
            persona_prompt = await self._get_active_persona_prompt(umo)
            if persona_prompt:
                parts.append(persona_prompt)

        custom_prompt = self._llm_custom_prompt()
        if custom_prompt:
            if parts:
                parts.append(
                    "以下是本插件额外的人格要求，请在保留当前人格风格的前提下遵守：\n"
                    + custom_prompt
                )
            else:
                parts.append(custom_prompt)

        if not parts:
            parts.append("你是一个盯人干活的监督机器人，说话直接、简短、带压迫感。")

        parts.append(self._llm_scene_prompt())
        return "\n\n".join(part for part in parts if part)

    def _resolve_provider(self, umo: str) -> Provider | None:
        provider_id = self._llm_provider_id()
        provider = None
        try:
            if provider_id:
                provider = self.context.get_provider_by_id(provider_id)
            else:
                provider = self.context.get_using_provider(umo=umo)
            if provider is None and provider_id:
                provider = self.context.get_using_provider(umo=umo)
        except Exception:
            provider = None
        return provider if isinstance(provider, Provider) else None

    def _safe_template(self, template: str, mapping: dict[str, str]) -> str:
        class SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        return template.format_map(SafeDict(mapping))

    def _build_fallback_reminder_text(
        self,
        task: dict[str, Any],
        todos: list[str],
        now: datetime,
    ) -> str:
        end_at = self._task_cycle_end(task, now)
        title = str(task.get("task_title") or "当前任务").strip()
        todo_text = "、".join(todos) if todos else "把手上的任务推进下去"
        mapping = {
            "name": str(task.get("target_user_name") or task.get("target_user_id") or "你"),
            "task": title,
            "todos": todo_text,
            "remaining": self._format_remaining(end_at, now) if end_at else "快去做。",
            "deadline": self._format_time(end_at),
            "creator": str(task.get("created_by_user_name") or "管理员"),
        }
        template = random.choice(self._fallback_templates())
        return self._safe_template(template, mapping)

    def _sanitize_llm_text(self, text: str) -> str:
        merged = str(text or "").strip()
        merged = re.sub(r"\s+", " ", merged)
        merged = merged.strip("` ")
        lowered = merged.lower()
        if merged and any(needle in lowered for needle in GENERIC_LLM_REPLY_NEEDLES):
            return ""
        if len(merged) > self._llm_max_output_chars():
            merged = merged[: self._llm_max_output_chars()].rstrip() + "…"
        return merged

    async def _build_reminder_text_for_session(
        self,
        task: dict[str, Any],
        todos: list[str],
        umo: str,
    ) -> str:
        now = self._now()
        fallback = self._build_fallback_reminder_text(task, todos, now)
        if not self._llm_enabled():
            return fallback

        provider = self._resolve_provider(umo)
        if provider is None:
            return fallback

        end_at = self._task_cycle_end(task, now)
        prompt_lines = [
            f"目标用户：{task.get('target_user_name') or task.get('target_user_id')}",
            f"监督发起人：{task.get('created_by_user_name') or task.get('created_by_user_id')}",
            f"任务标题：{task.get('task_title') or '未命名任务'}",
            f"当前时间：{self._format_time(now)}",
            f"结束时间：{self._format_time(end_at)}",
            f"剩余时间：{self._format_remaining(end_at, now) if end_at else '未知'}",
            f"期间提醒间隔：{self._format_duration_seconds(int(task.get('cooldown_seconds') or 0))}",
            f"提醒次数上限：{self._format_reminder_limit_text(int(task.get('reminder_limit') or 0))}",
            f"这次提醒重点待办：{'；'.join(todos) if todos else '没有单独待办，直接催这个任务'}",
            "请直接输出最终催促文本，不要自称模型，不要写“好的/当然”。",
        ]

        try:
            response = await provider.text_chat(
                system_prompt=await self._build_llm_system_prompt(umo),
                prompt="\n".join(prompt_lines),
            )
            text = self._sanitize_llm_text(
                str(getattr(response, "completion_text", "") or "")
            )
            return text or fallback
        except Exception as exc:
            logger.warning(
                f"WorkSupervisor LLM reminder generation failed, using fallback: {exc}",
                exc_info=True,
            )
            return fallback

    async def _build_reminder_text(
        self,
        task: dict[str, Any],
        todos: list[str],
        event: AstrMessageEvent,
    ) -> str:
        return await self._build_reminder_text_for_session(
            task,
            todos,
            str(event.unified_msg_origin or ""),
        )

    # ---- 监督提醒 ----
    def _pick_todos(self, task: dict[str, Any]) -> list[str]:
        try:
            todos = json.loads(str(task.get("todo_items_json") or "[]"))
        except Exception:
            todos = []
        if not isinstance(todos, list):
            todos = []
        items = [str(item).strip() for item in todos if str(item).strip()]
        if not items:
            return []
        pick_count = max(int(task.get("todo_pick_count") or 1), 1)
        pick_count = min(pick_count, len(items))
        if pick_count >= len(items):
            return items
        return random.sample(items, pick_count)

    def _task_reminder_count_exhausted(
        self,
        conn: sqlite3.Connection,
        task: dict[str, Any],
        now: datetime,
    ) -> bool:
        limit = max(int(task.get("reminder_limit") or 0), 0)
        if limit <= 0:
            return False
        since = None
        if self._normalize_schedule_kind(task.get("schedule_kind")) == "daily":
            since = self._task_cycle_start(task, now)
        sent = self._count_task_reminders(conn, int(task["id"]), since=since)
        return sent >= limit

    def _reminder_due(
        self,
        conn: sqlite3.Connection,
        task: dict[str, Any],
        now: datetime,
    ) -> bool:
        start_at = self._parse_dt(task.get("start_at"))
        if start_at and start_at > now:
            return False
        schedule_kind = self._normalize_schedule_kind(task.get("schedule_kind"))
        end_at = self._task_cycle_end(task, now)
        if schedule_kind == "once" and end_at and end_at <= now:
            return False
        if self._task_reminder_count_exhausted(conn, task, now):
            return False
        last_reminded_at = self._parse_dt(task.get("last_reminded_at"))
        if last_reminded_at is None:
            return True
        if schedule_kind == "daily":
            cycle_start = self._task_cycle_start(task, now)
            if cycle_start and last_reminded_at < cycle_start:
                return True
        cooldown_seconds = int(task.get("cooldown_seconds") or 0)
        return (now - last_reminded_at).total_seconds() >= cooldown_seconds

    def _render_task_status(self, task: dict[str, Any], now: datetime) -> str:
        start_at = self._parse_dt(task.get("start_at"))
        schedule_kind = self._normalize_schedule_kind(task.get("schedule_kind"))
        end_at = self._task_cycle_end(task, now)
        last_reminded_at = self._parse_dt(task.get("last_reminded_at"))
        try:
            todos = json.loads(str(task.get("todo_items_json") or "[]"))
        except Exception:
            todos = []
        todo_items = [str(item).strip() for item in todos if str(item).strip()]
        reminder_limit = max(int(task.get("reminder_limit") or 0), 0)
        reminder_sent = int(task.get("_reminder_sent") or 0)
        if reminder_limit > 0:
            reminder_limit_text = f"{reminder_sent}/{reminder_limit}"
        else:
            reminder_limit_text = self._format_reminder_limit_text(reminder_limit)
        if schedule_kind == "permanent":
            end_text = "不自动结束"
        elif schedule_kind == "daily":
            end_text = f"{self._format_time(end_at)}（每天重复）" if end_at else "每天重复"
        else:
            end_text = self._format_time(end_at)
        lines = [
            f"目标：{task.get('target_user_name') or task.get('target_user_id')}",
            f"任务：{task.get('task_title') or '未命名任务'}",
            f"状态：{task.get('status')}",
            f"开始时间：{self._format_time(start_at)}",
            f"持续时间：{self._format_schedule_duration_label(task)}",
            f"结束时间：{end_text}",
            f"期间提醒间隔：{self._format_duration_seconds(int(task.get('cooldown_seconds') or 0))}",
            f"提醒次数：{reminder_limit_text}",
            f"上次提醒：{self._format_time(last_reminded_at)}",
        ]
        if start_at and start_at > now and str(task.get("status")) == "active":
            lines.append(self._format_until_start(start_at, now))
        elif schedule_kind == "once" and end_at and str(task.get("status")) == "active":
            lines.append(self._format_remaining(end_at, now))
        elif schedule_kind == "daily" and end_at and str(task.get("status")) == "active":
            lines.append(self._format_remaining(end_at, now).replace("剩余时间", "本轮剩余时间", 1))
        if todo_items:
            lines.append("待办：")
            lines.extend(f"{index}. {item}" for index, item in enumerate(todo_items, start=1))
        return "\n".join(lines)

    # ---- 播报 ----
    def _kind_label(self, kind: str) -> str:
        return "更新内容" if kind == "update" else "内容预告"

    def _kind_feature_enabled(self, kind: str) -> bool:
        return self._update_feature_enabled() if kind == "update" else self._preview_feature_enabled()

    def _render_broadcast_text(self, kind: str, content: str) -> str:
        prefix = f"[{self._kind_label(kind)}]"
        body = str(content or "").strip()
        if body.startswith(prefix):
            return body
        return f"{prefix} {body}"

    async def _set_broadcast_job(
        self,
        event: AstrMessageEvent,
        *,
        kind: str,
        enabled: bool,
        time_hhmm: str,
        content: str,
    ) -> None:
        now = self._now()
        async with self._db_lock:
            with self._connect() as conn:
                self._upsert_broadcast_job(
                    conn,
                    kind=kind,
                    enabled=enabled,
                    session_id=event.unified_msg_origin,
                    session_label=str(event.get_group_id() or event.unified_msg_origin),
                    time_hhmm=time_hhmm,
                    content=content,
                    updated_by_user_id=str(event.get_sender_id() or ""),
                    updated_by_user_name=self._extract_sender_name(event),
                    now=now,
                )
                self._sync_broadcast_jobs_to_config_sync(conn)

    async def _toggle_broadcast_job(self, kind: str, enabled: bool) -> bool:
        now_iso = self._iso(self._now())
        async with self._db_lock:
            with self._connect() as conn:
                row = self._get_broadcast_job(conn, kind)
                if row is None:
                    return False
                conn.execute(
                    "UPDATE broadcast_jobs SET enabled = ?, updated_at = ? WHERE kind = ?",
                    (1 if enabled else 0, now_iso, kind),
                )
                conn.commit()
                self._sync_broadcast_jobs_to_config_sync(conn)
        return True

    async def _get_broadcast_status_text(self, kind: str) -> str:
        async with self._db_lock:
            with self._connect() as conn:
                row = self._get_broadcast_job(conn, kind)

        label = self._kind_label(kind)
        if row is None:
            return f"{label} 还没有设置。"

        last_sent_at = self._parse_dt(row.get("last_sent_at"))
        enabled_text = "开启" if int(row.get("enabled") or 0) else "关闭"
        feature_text = "允许" if self._kind_feature_enabled(kind) else "被配置禁用"
        return "\n".join(
            [
                f"{label} 功能状态：{enabled_text}",
                f"插件配置：{feature_text}",
                f"发送时间：{row.get('time_hhmm') or '未设置'}",
                f"目标会话：{row.get('session_label') or row.get('session_id') or '未设置'}",
                f"上次发送：{self._format_time(last_sent_at)}",
                f"文案：{row.get('content') or '未设置'}",
            ]
        )

    # ---- 事件钩子 ----
    async def _maybe_send_supervision_reminder(self, event: AstrMessageEvent) -> None:
        await self._sync_settings_tasks_from_config()
        if not self._enabled():
            return
        if self._ignore_self_messages() and event.get_sender_id() == event.get_self_id():
            return
        if not self._allow_private_chat() and self._is_private_chat(event):
            return
        if not self._event_group_allowed(event):
            return

        plain_text = self._extract_plain_text(event)
        if self._should_yield_to_normal_chat(event, plain_text):
            return
        if self._skip_message_for_commands(plain_text):
            return

        await self._ensure_scheduler_started()
        await self._expire_overdue_tasks()

        sender_id = str(event.get_sender_id() or "").strip()
        session_id = str(event.unified_msg_origin or "").strip()
        group_id = str(event.get_group_id() or "").strip()
        if not sender_id or not session_id:
            return

        now = self._now()
        async with self._db_lock:
            with self._connect() as conn:
                task = self._find_active_task_for_event(
                    conn,
                    sender_id,
                    session_id=session_id,
                    group_id=group_id,
                )
                if task and not self._reminder_due(conn, task, now):
                    task = None

        if not task:
            return

        task_id = int(task["id"])
        if task_id in self._inflight_task_ids:
            return

        self._inflight_task_ids.add(task_id)
        try:
            selected_todos = self._pick_todos(task)
            event.should_call_llm(False)
            reminder_text = await self._build_reminder_text(task, selected_todos, event)
            message_chain = MessageEventResult()
            if not self._is_private_chat(event):
                message_chain.at(
                    name=str(task.get("target_user_name") or task.get("target_user_id") or "你"),
                    qq=str(task.get("target_user_id") or event.get_sender_id()),
                ).message(" ")
            message_chain.message(reminder_text)
            event.set_result(message_chain.stop_event())
            async with self._db_lock:
                with self._connect() as conn:
                    self._touch_reminder(conn, task_id, reminder_text, event, now)
        except Exception as exc:
            logger.warning(f"WorkSupervisor failed to send reminder: {exc}", exc_info=True)
        finally:
            self._inflight_task_ids.discard(task_id)

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1000)
    async def on_message(self, event: AstrMessageEvent) -> None:
        if await self._maybe_handle_explicit_command(event):
            return
        await self._maybe_send_supervision_reminder(event)

    # ---- 监督命令 ----
    @filter.command_group("监督", alias={"督工"})
    def supervisor(self) -> None:
        """监督任务管理"""

    @supervisor.command("开始", alias={"创建", "新增"})
    async def start_supervision(
        self,
        event: AstrMessageEvent,
        payload: GreedyStr = "",
    ) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        await self._ensure_scheduler_started()

        gate_error = self._command_gate_error(event)
        if gate_error == "__ignore__":
            return
        if gate_error:
            yield event.plain_result(gate_error)
            return

        sender_id = str(event.get_sender_id() or "").strip()
        sender_name = self._extract_sender_name(event)
        mentions = self._extract_mentions(event)
        target_user_id = sender_id
        target_user_name = sender_name

        if mentions:
            target_user_id = mentions[0]["user_id"]
            target_user_name = mentions[0]["name"]

        cleaned_payload = self._clean_payload_after_mentions(
            self._resolve_command_payload(
                event,
                payload,
                "监督 开始",
                "监督 创建",
                "监督 新增",
                "督工 开始",
                "督工 创建",
                "督工 新增",
            )
        )

        parsed = self._parse_start_payload(cleaned_payload)
        if parsed is None:
            yield event.plain_result(
                "格式不对。标准格式：\n"
                "监督 开始 任务=写第一章 待办=大纲、正文、校对 时长=3h 冷却=2h 抽取=3\n"
                "监督 开始 @小明 任务=做海报 待办=出图、排版 时长=2h 冷却=1h"
            )
            return

        result = await self._create_supervision_result(
            event,
            target_user_id=target_user_id,
            target_user_name=target_user_name,
            parsed=parsed,
        )
        yield event.plain_result(result)

    @supervisor.command("状态", alias={"查看", "查询"})
    async def status_supervision(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)

        mentions = self._extract_mentions(event)
        target_user_id = mentions[0]["user_id"] if mentions else str(event.get_sender_id() or "").strip()
        result = await self._status_supervision_result(
            event,
            target_user_id=target_user_id,
        )
        yield event.plain_result(result)

    @supervisor.command("完成", alias={"结束", "done"})
    async def complete_supervision(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)

        sender_id = str(event.get_sender_id() or "").strip()
        mentions = self._extract_mentions(event)
        target_user_id = mentions[0]["user_id"] if mentions else sender_id
        result = await self._complete_supervision_result(
            event,
            target_user_id=target_user_id,
        )
        if result is None:
            return
        yield event.plain_result(result)

    @supervisor.command("取消", alias={"abort"})
    async def cancel_supervision(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        await self._sync_settings_tasks_from_config()
        await self._ensure_scheduler_started()
        await self._expire_overdue_tasks()

        sender_id = str(event.get_sender_id() or "").strip()
        mentions = self._extract_mentions(event)
        target_user_id = mentions[0]["user_id"] if mentions else sender_id

        async with self._db_lock:
            with self._connect() as conn:
                task = self._find_active_task_by_target(conn, target_user_id)
                if task is None:
                    yield event.plain_result("没有找到进行中的监督任务。")
                    return
                if sender_id not in {str(task["target_user_id"]), str(task["created_by_user_id"])} and not self._is_admin(event):
                    yield event.plain_result("只有目标本人、任务创建者或管理员可以取消这个监督任务。")
                    return
                self._mark_task_status(conn, int(task["id"]), "cancelled", self._now())
                self._sync_active_tasks_to_settings_sync(conn)

        yield event.plain_result(f"已取消监督：{task['target_user_name']} 的《{task['task_title']}》")

    @supervisor.command("帮助", alias={"help"})
    async def help_supervision(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        yield event.plain_result(self._supervision_help_text())

    # ---- 更新内容命令 ----
    @filter.command_group("更新内容")
    def update_content(self) -> None:
        """更新内容播报"""

    @update_content.command("设置", alias={"设定"})
    async def update_set(
        self,
        event: AstrMessageEvent,
        payload: GreedyStr = "",
    ) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        await self._ensure_scheduler_started()
        if not self._is_private_chat(event) and not self._is_admin(event):
            yield event.plain_result("群聊里只有管理员可以设置更新内容播报。")
            return

        parsed = self._parse_broadcast_setting(
            self._resolve_command_payload(
                event,
                payload,
                "更新内容 设置",
                "更新内容 设定",
            )
        )
        if parsed is None:
            yield event.plain_result("格式不对。标准格式：更新内容 设置 时间=21:00 内容=今天更新了第一章和封面")
            return

        hhmm, content = parsed
        await self._set_broadcast_job(
            event,
            kind="update",
            enabled=True,
            time_hhmm=hhmm,
            content=content,
        )
        yield event.plain_result(f"已设置更新内容播报：每天 {hhmm} 发送到当前会话。")

    @update_content.command("开")
    async def update_on(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        if not self._is_private_chat(event) and not self._is_admin(event):
            yield event.plain_result("群聊里只有管理员可以开启更新内容播报。")
            return
        ok = await self._toggle_broadcast_job("update", True)
        yield event.plain_result("已开启更新内容播报。" if ok else "还没有设置更新内容播报，请先执行“更新内容 设置 ...”。")

    @update_content.command("关")
    async def update_off(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        if not self._is_private_chat(event) and not self._is_admin(event):
            yield event.plain_result("群聊里只有管理员可以关闭更新内容播报。")
            return
        ok = await self._toggle_broadcast_job("update", False)
        yield event.plain_result("已关闭更新内容播报。" if ok else "还没有设置更新内容播报。")

    @update_content.command("状态")
    async def update_status(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        yield event.plain_result(await self._get_broadcast_status_text("update"))

    @update_content.command("立即发送", alias={"发送"})
    async def update_send_now(
        self,
        event: AstrMessageEvent,
        payload: GreedyStr = "",
    ) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        if not self._is_private_chat(event) and not self._is_admin(event):
            yield event.plain_result("群聊里只有管理员可以立即发送更新内容。")
            return

        content = str(payload or "").strip()
        if not content:
            async with self._db_lock:
                with self._connect() as conn:
                    row = self._get_broadcast_job(conn, "update")
            if row is None or not str(row.get("content") or "").strip():
                yield event.plain_result("没有找到已保存的更新内容文案。")
                return
            content = str(row["content"]).strip()

        await event.send(MessageChain().message(self._render_broadcast_text("update", content)))
        yield event.plain_result("更新内容已发送。")

    # ---- 内容预告命令 ----
    @filter.command_group("内容预告")
    def preview_content(self) -> None:
        """内容预告播报"""

    @preview_content.command("设置", alias={"设定"})
    async def preview_set(
        self,
        event: AstrMessageEvent,
        payload: GreedyStr = "",
    ) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        await self._ensure_scheduler_started()
        if not self._is_private_chat(event) and not self._is_admin(event):
            yield event.plain_result("群聊里只有管理员可以设置内容预告播报。")
            return

        parsed = self._parse_broadcast_setting(
            self._resolve_command_payload(
                event,
                payload,
                "内容预告 设置",
                "内容预告 设定",
            )
        )
        if parsed is None:
            yield event.plain_result("格式不对。标准格式：内容预告 设置 时间=20:00 内容=明天预告第二章和设定图")
            return

        hhmm, content = parsed
        await self._set_broadcast_job(
            event,
            kind="preview",
            enabled=True,
            time_hhmm=hhmm,
            content=content,
        )
        yield event.plain_result(f"已设置内容预告播报：每天 {hhmm} 发送到当前会话。")

    @preview_content.command("开")
    async def preview_on(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        if not self._is_private_chat(event) and not self._is_admin(event):
            yield event.plain_result("群聊里只有管理员可以开启内容预告播报。")
            return
        ok = await self._toggle_broadcast_job("preview", True)
        yield event.plain_result("已开启内容预告播报。" if ok else "还没有设置内容预告播报，请先执行“内容预告 设置 ...”。")

    @preview_content.command("关")
    async def preview_off(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        if not self._is_private_chat(event) and not self._is_admin(event):
            yield event.plain_result("群聊里只有管理员可以关闭内容预告播报。")
            return
        ok = await self._toggle_broadcast_job("preview", False)
        yield event.plain_result("已关闭内容预告播报。" if ok else "还没有设置内容预告播报。")

    @preview_content.command("状态")
    async def preview_status(self, event: AstrMessageEvent) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        yield event.plain_result(await self._get_broadcast_status_text("preview"))

    @preview_content.command("立即发送", alias={"发送"})
    async def preview_send_now(
        self,
        event: AstrMessageEvent,
        payload: GreedyStr = "",
    ) -> AsyncGenerator[MessageEventResult, None]:
        event.should_call_llm(False)
        if not self._is_private_chat(event) and not self._is_admin(event):
            yield event.plain_result("群聊里只有管理员可以立即发送内容预告。")
            return

        content = str(payload or "").strip()
        if not content:
            async with self._db_lock:
                with self._connect() as conn:
                    row = self._get_broadcast_job(conn, "preview")
            if row is None or not str(row.get("content") or "").strip():
                yield event.plain_result("没有找到已保存的内容预告文案。")
                return
            content = str(row["content"]).strip()

        await event.send(MessageChain().message(self._render_broadcast_text("preview", content)))
        yield event.plain_result("内容预告已发送。")
