from __future__ import annotations

import asyncio
import importlib.util
import os
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Run the smoke test against the local AstrBot runtime so message and provider
# interfaces stay aligned with the currently installed desktop version.
ROOT = Path(__file__).resolve().parents[1]
ASTRBOT_APP = Path(
    os.environ.get(
        "ASTRBOT_APP",
        r"C:\Users\Administrator\AstrBotDesktopTest\backend\app",
    )
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ASTRBOT_APP) not in sys.path:
    sys.path.insert(0, str(ASTRBOT_APP))

from astrbot.api.event import MessageEventResult
from astrbot.api.message_components import At, Plain
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.provider import Provider


def load_plugin_main():
    module_path = ROOT / "main.py"
    spec = importlib.util.spec_from_file_location(
        "work_supervisor_plugin_main",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load plugin module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plugin_main = load_plugin_main()


def plain_text_of(result: MessageEventResult) -> str:
    return result.get_plain_text(with_other_comps_mark=True).strip()


def plain_texts_of_chain(chain: Any) -> list[str]:
    return [comp.text for comp in getattr(chain, "chain", []) if isinstance(comp, Plain)]


def latest_sent_plain_text(event: Any) -> str:
    return " ".join(plain_texts_of_chain(event.sent_messages[-1])).strip()


class DummyProvider(Provider):
    def __init__(self) -> None:
        super().__init__(
            provider_config={"id": "dummy", "type": "dummy"},
            provider_settings={},
        )
        self.set_model("dummy-model")
        self.calls: list[dict[str, str]] = []
        self.completion_text = "别磨蹭，先把出图和排版往前推。"

    def get_current_key(self) -> str:
        return "dummy-key"

    def set_key(self, key: str) -> None:
        return None

    async def get_models(self) -> list[str]:
        return ["dummy-model"]

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.calls.append(
            {
                "system_prompt": str(kwargs.get("system_prompt") or ""),
                "prompt": str(kwargs.get("prompt") or ""),
            }
        )
        return LLMResponse(
            role="assistant",
            completion_text=self.completion_text,
        )


class DummyPersonaManager:
    async def resolve_selected_persona(self, **kwargs):
        return None, {"prompt": "你是一个说话直接的毒舌女仆。"}, None, None

    async def get_default_persona_v3(self, **kwargs):
        return {"prompt": "默认人格"}


class DummyConversationManager:
    async def get_curr_conversation_id(self, umo: str):
        return None

    async def get_conversation(self, **kwargs):
        return None


class DummyContext:
    def __init__(self, provider: DummyProvider) -> None:
        self.provider = provider
        self.persona_manager = DummyPersonaManager()
        self.conversation_manager = DummyConversationManager()
        self.sent_messages: list[dict[str, Any]] = []

    def get_config(self, umo: str | None = None) -> dict[str, Any]:
        return {"provider_settings": {}}

    def get_provider_by_id(self, provider_id: str):
        if provider_id in {"", "dummy"}:
            return self.provider
        return None

    def get_using_provider(self, umo: str | None = None):
        return self.provider

    async def send_message(self, session: str, message_chain: Any) -> bool:
        self.sent_messages.append(
            {
                "session": session,
                "plain_texts": plain_texts_of_chain(message_chain),
            }
        )
        return True


@dataclass
class FakeEvent:
    sender_id: str
    sender_name: str
    unified_msg_origin: str
    message_str: str
    messages: list[Any]
    group_id: str = ""
    self_id: str = "bot"
    private: bool = False
    admin: bool = False
    is_at_or_wake_command: bool = False

    def __post_init__(self) -> None:
        self.sent_messages: list[Any] = []
        self.call_llm = True
        self.stopped = False
        self.result: MessageEventResult | None = None

    def get_sender_id(self) -> str:
        return self.sender_id

    def get_self_id(self) -> str:
        return self.self_id

    def get_group_id(self) -> str:
        return self.group_id

    def get_sender_name(self) -> str:
        return self.sender_name

    def get_sender_nickname(self) -> str:
        return self.sender_name

    def get_messages(self) -> list[Any]:
        return self.messages

    def get_message_str(self) -> str:
        return self.message_str

    def is_private_chat(self) -> bool:
        return self.private

    def is_admin(self) -> bool:
        return self.admin

    def should_call_llm(self, call_llm: bool) -> None:
        self.call_llm = call_llm

    def stop_event(self) -> None:
        self.stopped = True

    def plain_result(self, text: str) -> MessageEventResult:
        return MessageEventResult().message(text)

    def set_result(self, result: MessageEventResult | str) -> None:
        if isinstance(result, str):
            result = MessageEventResult().message(result)
        self.result = result
        if result.is_stopped():
            self.stopped = True
        self.sent_messages.append(result)

    async def send(self, message_chain: Any) -> None:
        self.sent_messages.append(message_chain)


async def collect_results(async_gen) -> list[str]:
    results: list[str] = []
    async for item in async_gen:
        results.append(plain_text_of(item))
    return results


async def run_smoke_test() -> list[str]:
    random.seed(0)
    original_get_data_dir = plugin_main.StarTools.__dict__["get_data_dir"]
    temp_root = ROOT / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / "local_smoke_runtime"

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        data_dir = temp_dir
        plugin_main.StarTools.get_data_dir = classmethod(
            lambda cls, plugin_name=None: data_dir
        )

        provider = DummyProvider()
        context = DummyContext(provider)
        plugin = plugin_main.WorkSupervisorPlugin(
            context,
            config={
                "enabled": True,
                "allow_private_chat": True,
                "allow_supervise_others": True,
                "allow_non_admin_supervise_others": False,
                "settings_tasks": [],
                "broadcast_settings": [],
                "default_duration_minutes": 180,
                "default_cooldown_minutes": 120,
                "default_todo_pick_count": 3,
                "fallback_reminder_templates": (
                    "{name}，别摸了，`{task}` 还挂着。先去把这些做掉：{todos}。{remaining}\n"
                    "{name}，现在不是装死的时候。`{task}` 还没完，先处理：{todos}。{remaining}"
                ),
                "llm_enabled": True,
                "llm_follow_active_persona": True,
                "llm_provider_id": "",
                "update_feature_enabled": True,
                "preview_feature_enabled": True,
            },
        )

        async def no_scheduler() -> None:
            return None

        plugin._ensure_scheduler_started = no_scheduler  # type: ignore[method-assign]

        passed: list[str] = []
        assert plugin._skip_message_for_commands("监督 状态")
        assert plugin._skip_message_for_commands("督工 状态")
        passed.append("command_prefix_aliases")

        direct_help_event = FakeEvent(
            sender_id="u130",
            sender_name="HelpUser",
            unified_msg_origin="aiocqhttp:FriendMessage:u130",
            message_str="监督",
            messages=[Plain("监督")],
            private=True,
        )
        await plugin.on_message(direct_help_event)
        assert len(direct_help_event.sent_messages) == 1
        assert "监督命令" in plain_text_of(direct_help_event.sent_messages[0])
        assert direct_help_event.call_llm is False
        assert direct_help_event.stopped is True
        passed.append("direct_help_command")

        direct_private_start_event = FakeEvent(
            sender_id="u131",
            sender_name="DirectUser",
            unified_msg_origin="aiocqhttp:FriendMessage:u131",
            message_str="监督 开始 @DirectUser 任务=写周报 待办=整理数据、写摘要 时长=2h 冷却=1h",
            messages=[
                Plain("监督 开始 "),
                At(name="DirectUser", qq="u131"),
                Plain(" 任务=写周报 待办=整理数据、写摘要 时长=2h 冷却=1h"),
            ],
            private=True,
        )
        await plugin.on_message(direct_private_start_event)
        assert len(direct_private_start_event.sent_messages) == 1
        assert "已开始监督" in plain_text_of(direct_private_start_event.sent_messages[0])
        assert "写周报" in plain_text_of(direct_private_start_event.sent_messages[0])
        assert direct_private_start_event.call_llm is False
        assert direct_private_start_event.stopped is True
        passed.append("direct_private_start_command")

        direct_at_bot_start_event = FakeEvent(
            sender_id="u132",
            sender_name="AtBotUser",
            unified_msg_origin="aiocqhttp:GroupMessage:g132",
            message_str="@bot 监督 开始 @AtBotUser 任务=做海报 待办=出图、排版 时长=2h 冷却=1h",
            messages=[
                At(name="SupervisorBot", qq="bot"),
                Plain(" 监督 开始 "),
                At(name="AtBotUser", qq="u132"),
                Plain(" 任务=做海报 待办=出图、排版 时长=2h 冷却=1h"),
            ],
            group_id="g132",
            private=False,
        )
        await plugin.on_message(direct_at_bot_start_event)
        assert len(direct_at_bot_start_event.sent_messages) == 1
        assert "已开始监督" in plain_text_of(direct_at_bot_start_event.sent_messages[0])
        assert "做海报" in plain_text_of(direct_at_bot_start_event.sent_messages[0])
        assert direct_at_bot_start_event.call_llm is False
        assert direct_at_bot_start_event.stopped is True
        passed.append("direct_at_bot_start_command")

        raw_message_fallback_event = FakeEvent(
            sender_id="u133",
            sender_name="RawOnlyUser",
            unified_msg_origin="aiocqhttp:GroupMessage:g133",
            message_str="监督 开始 @RawOnlyUser(133) 任务=做海报 待办=出图、排版 时长=2h 冷却=1h",
            messages=[
                At(name="RawOnlyUser", qq="u133"),
                Plain("监督"),
            ],
            group_id="g133",
            private=False,
            admin=True,
        )
        await plugin.on_message(raw_message_fallback_event)
        assert len(raw_message_fallback_event.sent_messages) == 1
        assert "已开始监督" in plain_text_of(raw_message_fallback_event.sent_messages[0])
        assert "做海报" in plain_text_of(raw_message_fallback_event.sent_messages[0])
        assert raw_message_fallback_event.call_llm is False
        assert raw_message_fallback_event.stopped is True
        passed.append("raw_message_command_fallback")

        self_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str="监督 开始 任务=写第一章 待办=大纲、正文、校对 时长=3h 冷却=2h 抽取=3",
            messages=[Plain("监督 开始 任务=写第一章 待办=大纲、正文、校对 时长=3h 冷却=2h 抽取=3")],
            private=True,
        )
        create_self = await collect_results(
            plugin.start_supervision(
                self_event,
                "任务=写第一章 待办=大纲、正文、校对 时长=3h 冷却=2h 抽取=3",
            )
        )
        assert create_self and "写第一章" in create_self[0]
        assert any(
            item.get("task_title") == "写第一章"
            for item in plugin.config.get("settings_tasks", [])
        )
        passed.append("self_start")

        second_self_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str="监督 开始 任务=做海报 待办=出图、排版 时长=2h 冷却=1h",
            messages=[Plain("监督 开始 任务=做海报 待办=出图、排版 时长=2h 冷却=1h")],
            private=True,
        )
        create_self_second = await collect_results(
            plugin.start_supervision(
                second_self_event,
                "任务=做海报 待办=出图、排版 时长=2h 冷却=1h",
            )
        )
        assert create_self_second and "做海报" in create_self_second[0]
        with plugin._connect() as conn:
            self_tasks = plugin._list_active_tasks_by_target(conn, "u100")
        assert len(self_tasks) == 2
        self_tasks_by_title = {str(task["task_title"]): int(task["id"]) for task in self_tasks}
        first_self_task_id = self_tasks_by_title["写第一章"]
        second_self_task_id = self_tasks_by_title["做海报"]
        passed.append("self_start_second_task")

        plugin.config["normal_chat_yield_prefixes"] = ["/", "！"]

        mention_bot_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:GroupMessage:g200",
            message_str="@bot 帮我看看今天该先做什么",
            messages=[At(name="SupervisorBot", qq="bot"), Plain(" 帮我看看今天该先做什么")],
            group_id="g200",
            private=False,
        )
        await plugin.on_message(mention_bot_event)
        assert len(mention_bot_event.sent_messages) == 0
        assert mention_bot_event.call_llm is True
        assert mention_bot_event.stopped is False
        passed.append("mention_bot_yields_to_normal_chat")

        slash_command_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str="/chat 帮我规划一下今天",
            messages=[Plain("/chat 帮我规划一下今天")],
            private=True,
        )
        await plugin.on_message(slash_command_event)
        assert len(slash_command_event.sent_messages) == 0
        assert slash_command_event.call_llm is True
        assert slash_command_event.stopped is False
        passed.append("slash_command_yields_to_normal_chat")

        fullwidth_bang_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str="！今天安排一下",
            messages=[Plain("！今天安排一下")],
            private=True,
        )
        await plugin.on_message(fullwidth_bang_event)
        assert len(fullwidth_bang_event.sent_messages) == 0
        assert fullwidth_bang_event.call_llm is True
        assert fullwidth_bang_event.stopped is False
        passed.append("custom_yield_prefix")

        self_status_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str="监督 状态",
            messages=[Plain("监督 状态")],
            private=True,
        )
        self_status = await collect_results(plugin.status_supervision(self_status_event))
        assert self_status and "当前共有 2 个进行中的监督任务" in self_status[0]
        assert f"#{first_self_task_id}" in self_status[0]
        assert f"#{second_self_task_id}" in self_status[0]
        passed.append("self_status_multi_task_list")

        self_selected_status_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str=f"监督 状态 #{first_self_task_id}",
            messages=[Plain(f"监督 状态 #{first_self_task_id}")],
            private=True,
        )
        self_selected_status = await collect_results(
            plugin.status_supervision(self_selected_status_event, f"#{first_self_task_id}")
        )
        assert self_selected_status and "写第一章" in self_selected_status[0] and "active" in self_selected_status[0]
        passed.append("self_status_selected_task")

        self_cross_session_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:GroupMessage:g200",
            message_str="cross session ping",
            messages=[Plain("cross session ping")],
            group_id="g200",
            private=False,
        )
        await plugin.on_message(self_cross_session_event)
        assert len(self_cross_session_event.sent_messages) == 1
        assert self_cross_session_event.call_llm is False
        assert self_cross_session_event.stopped is True
        passed.append("private_task_cross_session_reminder")

        self_complete_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str=f"监督 完成 #{first_self_task_id}",
            messages=[Plain(f"监督 完成 #{first_self_task_id}")],
            private=True,
        )
        self_done = await collect_results(
            plugin.complete_supervision(self_complete_event, f"#{first_self_task_id}")
        )
        assert self_done and "写第一章" in self_done[0]
        assert not any(
            item.get("task_title") == "写第一章"
            for item in plugin.config.get("settings_tasks", [])
        )
        passed.append("self_complete")

        remaining_status_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str="监督 状态",
            messages=[Plain("监督 状态")],
            private=True,
        )
        remaining_status = await collect_results(plugin.status_supervision(remaining_status_event))
        assert remaining_status and "做海报" in remaining_status[0]
        assert "写第一章" not in remaining_status[0]
        passed.append("self_status_after_selected_complete")

        self_cancel_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str="监督 取消 任务=做海报",
            messages=[Plain("监督 取消 任务=做海报")],
            private=True,
        )
        self_cancel = await collect_results(
            plugin.cancel_supervision(self_cancel_event, "任务=做海报")
        )
        assert self_cancel and "做海报" in self_cancel[0]
        passed.append("self_cancel_selected_task")

        no_task = await collect_results(plugin.status_supervision(remaining_status_event))
        assert no_task and "没有" in no_task[0]
        passed.append("self_status_after_complete")

        settings_start_at = plugin._now() - plugin_main.timedelta(minutes=45)
        settings_deadline_at = settings_start_at + plugin_main.timedelta(hours=2)
        plugin.config["settings_tasks"] = [
            {
                "__template_key": "supervision_task",
                "enabled": True,
                "platform_id": "aiocqhttp",
                "session_type": "FriendMessage",
                "target_user_id": "u300",
                "target_user_name": "ConfigUser",
                "task_title": "设置页任务",
                "todo_items": "拆需求\n写代码",
                "start_at": plugin._format_time(settings_start_at),
                "duration": "2h",
                "end_at": plugin._format_time(settings_deadline_at),
                "reminder_interval": "30m",
                "reminder_count": 2,
            }
        ]
        await plugin._sync_settings_tasks_from_config(force=True)
        config_event = FakeEvent(
            sender_id="u300",
            sender_name="ConfigUser",
            unified_msg_origin="aiocqhttp:FriendMessage:u300",
            message_str="监督 状态",
            messages=[Plain("监督 状态")],
            private=True,
        )
        config_status = await collect_results(plugin.status_supervision(config_event))
        assert config_status and "设置页任务" in config_status[0]
        assert plugin._format_time(settings_start_at) in config_status[0]
        assert "期间提醒间隔：30 分钟" in config_status[0]
        assert "提醒次数：0/2" in config_status[0]
        assert any(
            item.get("task_title") == "设置页任务"
            and item.get("session_id") == "aiocqhttp:FriendMessage:u300"
            and item.get("start_at") == plugin._format_time(settings_start_at)
            and item.get("end_at") == plugin._format_time(settings_deadline_at)
            and item.get("reminder_interval") == "30m"
            and item.get("reminder_count") == 2
            and item.get("task_key")
            for item in plugin.config.get("settings_tasks", [])
        )
        passed.append("settings_task_import")

        multi_settings_start_at = plugin._now() - plugin_main.timedelta(minutes=20)
        plugin.config["settings_tasks"] = [
            {
                "__template_key": "supervision_task",
                "enabled": True,
                "platform_id": "aiocqhttp",
                "session_type": "FriendMessage",
                "target_user_id": "u305",
                "target_user_name": "MultiConfigUser",
                "task_title": "设置页多任务A",
                "todo_items": "A1\nA2",
                "start_at": plugin._format_time(multi_settings_start_at),
                "duration": "1h",
                "reminder_interval": "15m",
            },
            {
                "__template_key": "supervision_task",
                "enabled": True,
                "platform_id": "aiocqhttp",
                "session_type": "FriendMessage",
                "target_user_id": "u305",
                "target_user_name": "MultiConfigUser",
                "task_title": "设置页多任务B",
                "todo_items": "B1\nB2",
                "start_at": plugin._format_time(multi_settings_start_at),
                "duration": "2h",
                "reminder_interval": "20m",
            },
        ]
        await plugin._sync_settings_tasks_from_config(force=True)
        with plugin._connect() as conn:
            multi_settings_tasks = plugin._list_active_tasks_by_target(conn, "u305")
        assert len(multi_settings_tasks) == 2
        assert {str(task["task_title"]) for task in multi_settings_tasks} == {"设置页多任务A", "设置页多任务B"}
        assert sum(
            1
            for item in plugin.config.get("settings_tasks", [])
            if item.get("target_user_id") == "u305"
        ) == 2
        plugin.config["settings_tasks"] = [
            {
                "__template_key": "supervision_task",
                "enabled": True,
                "platform_id": "aiocqhttp",
                "session_type": "FriendMessage",
                "target_user_id": "u300",
                "target_user_name": "ConfigUser",
                "task_title": "设置页任务",
                "todo_items": "拆需求\n写代码",
                "start_at": plugin._format_time(settings_start_at),
                "duration": "2h",
                "end_at": plugin._format_time(settings_deadline_at),
                "reminder_interval": "30m",
                "reminder_count": 2,
            }
        ]
        await plugin._sync_settings_tasks_from_config(force=True)
        passed.append("settings_task_import_multi_same_target")

        await plugin._run_due_settings_initial_reminders()
        assert len(context.sent_messages) == 1
        assert context.sent_messages[0]["session"] == "aiocqhttp:FriendMessage:u300"
        context.sent_messages.clear()
        passed.append("settings_task_initial_push")

        config_done = await collect_results(plugin.complete_supervision(config_event))
        assert config_done and "设置页任务" in config_done[0]
        assert not any(
            item.get("task_title") == "设置页任务"
            for item in plugin.config.get("settings_tasks", [])
        )
        passed.append("settings_task_complete_sync")

        legacy_start_at = plugin._now() - plugin_main.timedelta(minutes=30)
        legacy_deadline_at = legacy_start_at + plugin_main.timedelta(hours=1)
        plugin.config["settings_tasks"] = [
            {
                "__template_key": "supervision_task",
                "enabled": True,
                "platform_id": "aiocqhttp",
                "session_type": "FriendMessage",
                "target_user_id": "u304",
                "target_user_name": "LegacyUser",
                "task_title": "legacy-task",
                "todo_items": "legacy-todo",
                "start_at": plugin._format_time(legacy_start_at),
                "duration": "1h",
                "deadline_at": plugin._format_time(legacy_deadline_at),
                "cooldown": "15m",
                "todo_pick_count": 1,
            }
        ]
        await plugin._sync_settings_tasks_from_config(force=True)
        assert any(
            item.get("task_title") == "legacy-task"
            and item.get("end_at") == plugin._format_time(legacy_deadline_at)
            and item.get("reminder_interval") == "15m"
            and "deadline_at" not in item
            and "cooldown" not in item
            for item in plugin.config.get("settings_tasks", [])
        )
        plugin.config["settings_tasks"] = []
        await plugin._sync_settings_tasks_from_config(force=True)
        passed.append("legacy_settings_normalized")

        future_start_at = plugin._now() + plugin_main.timedelta(minutes=20)
        plugin.config["settings_tasks"] = [
            {
                "__template_key": "supervision_task",
                "enabled": True,
                "platform_id": "aiocqhttp",
                "session_type": "FriendMessage",
                "target_user_id": "u301",
                "target_user_name": "FutureUser",
                "task_title": "未来开始任务",
                "todo_items": "先别催",
                "start_at": plugin._format_time(future_start_at),
                "duration": "1h",
                "reminder_interval": "10m",
                "reminder_count": 1,
            }
        ]
        await plugin._sync_settings_tasks_from_config(force=True)
        future_event = FakeEvent(
            sender_id="u301",
            sender_name="FutureUser",
            unified_msg_origin="aiocqhttp:FriendMessage:u301",
            message_str="我先路过一下",
            messages=[Plain("我先路过一下")],
            private=True,
        )
        await plugin.on_message(future_event)
        assert future_event.call_llm is True
        assert future_event.stopped is False
        assert len(future_event.sent_messages) == 0
        future_status = await collect_results(plugin.status_supervision(future_event))
        assert future_status and "距离开始" in future_status[0]
        plugin.config["settings_tasks"] = []
        await plugin._sync_settings_tasks_from_config(force=True)
        passed.append("future_start_waits")

        daily_start_at = plugin._now() - plugin_main.timedelta(minutes=5)
        plugin.config["settings_tasks"] = [
            {
                "__template_key": "supervision_task",
                "enabled": True,
                "platform_id": "aiocqhttp",
                "session_type": "FriendMessage",
                "target_user_id": "u302",
                "target_user_name": "DailyUser",
                "task_title": "每天任务",
                "todo_items": "打卡",
                "start_at": plugin._format_time(daily_start_at),
                "duration": "每天",
                "reminder_interval": "0m",
                "reminder_count": 1,
            }
        ]
        await plugin._sync_settings_tasks_from_config(force=True)
        daily_status_event = FakeEvent(
            sender_id="u302",
            sender_name="DailyUser",
            unified_msg_origin="aiocqhttp:FriendMessage:u302",
            message_str="监督 状态",
            messages=[Plain("监督 状态")],
            private=True,
        )
        daily_status = await collect_results(plugin.status_supervision(daily_status_event))
        assert daily_status and "持续时间：每天" in daily_status[0]
        assert any(
            item.get("task_title") == "每天任务" and item.get("duration") == "每天"
            for item in plugin.config.get("settings_tasks", [])
        )
        plugin.config["settings_tasks"] = []
        await plugin._sync_settings_tasks_from_config(force=True)
        passed.append("daily_duration_keyword")

        permanent_start_at = plugin._now() - plugin_main.timedelta(minutes=5)
        plugin.config["settings_tasks"] = [
            {
                "__template_key": "supervision_task",
                "enabled": True,
                "platform_id": "aiocqhttp",
                "session_type": "FriendMessage",
                "target_user_id": "u303",
                "target_user_name": "ForeverUser",
                "task_title": "永久任务",
                "todo_items": "持续推进",
                "start_at": plugin._format_time(permanent_start_at),
                "duration": "永久",
                "reminder_interval": "0m",
                "reminder_count": 1,
            }
        ]
        await plugin._sync_settings_tasks_from_config(force=True)
        permanent_event = FakeEvent(
            sender_id="u303",
            sender_name="ForeverUser",
            unified_msg_origin="aiocqhttp:FriendMessage:u303",
            message_str="我来一次",
            messages=[Plain("我来一次")],
            private=True,
        )
        await plugin.on_message(permanent_event)
        assert len(permanent_event.sent_messages) == 1
        permanent_event_second = FakeEvent(
            sender_id="u303",
            sender_name="ForeverUser",
            unified_msg_origin="aiocqhttp:FriendMessage:u303",
            message_str="我再来一次",
            messages=[Plain("我再来一次")],
            private=True,
        )
        await plugin.on_message(permanent_event_second)
        assert len(permanent_event_second.sent_messages) == 0
        permanent_status = await collect_results(plugin.status_supervision(permanent_event_second))
        assert permanent_status and "持续时间：永久" in permanent_status[0]
        assert "提醒次数：1/1" in permanent_status[0]
        plugin.config["settings_tasks"] = []
        await plugin._sync_settings_tasks_from_config(force=True)
        passed.append("permanent_duration_limit")

        plain_supervision_phrase_event = FakeEvent(
            sender_id="u102",
            sender_name="Nina",
            unified_msg_origin="aiocqhttp:FriendMessage:u102",
            message_str="接下来两小时监督我写周报，待办是整理数据、写摘要、发群里，冷却一小时",
            messages=[Plain("接下来两小时监督我写周报，待办是整理数据、写摘要、发群里，冷却一小时")],
            private=True,
        )
        await plugin.on_message(plain_supervision_phrase_event)
        assert plain_supervision_phrase_event.call_llm is True
        assert plain_supervision_phrase_event.stopped is False
        assert len(plain_supervision_phrase_event.sent_messages) == 0
        passed.append("plain_supervision_phrase_ignored")

        full_text_fallback_event = FakeEvent(
            sender_id="u101",
            sender_name="Eve",
            unified_msg_origin="aiocqhttp:FriendMessage:u101",
            message_str="监督 开始 任务=写周报 待办=整理数据、写摘要、发群里 时长=2h 冷却=1h",
            messages=[
                Plain("监督 开始 任务=写周报 待办=整理数据、写摘要、发群里 时长=2h 冷却=1h")
            ],
            private=True,
        )
        create_from_full_text = await collect_results(
            plugin.start_supervision(full_text_fallback_event, "任务=写周报")
        )
        assert create_from_full_text and "2 小时" in create_from_full_text[0]
        assert "1 小时" in create_from_full_text[0]
        assert "整理数据" in create_from_full_text[0]
        passed.append("full_text_payload_fallback")

        full_text_done_event = FakeEvent(
            sender_id="u101",
            sender_name="Eve",
            unified_msg_origin="aiocqhttp:FriendMessage:u101",
            message_str="监督 完成",
            messages=[Plain("监督 完成")],
            private=True,
        )
        done_from_full_text = await collect_results(
            plugin.complete_supervision(full_text_done_event)
        )
        assert done_from_full_text and "写周报" in done_from_full_text[0]

        provider.completion_text = (
            "I am ready to help with the task and can use the available tools to make progress."
        )
        placeholder_text = await plugin._build_reminder_text(
            {
                "target_user_id": "u101",
                "target_user_name": "Eve",
                "created_by_user_id": "u999",
                "created_by_user_name": "Boss",
                "task_title": "写周报",
                "end_at": plugin._iso(plugin._now() + plugin_main.timedelta(hours=2)),
                "cooldown_seconds": 60,
            },
            ["整理数据", "写摘要"],
            full_text_fallback_event,
        )
        assert "写周报" in placeholder_text and "整理数据" in placeholder_text
        passed.append("placeholder_fallback")
        provider.completion_text = "别磨蹭，先把出图和排版往前推。"

        non_admin_group_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:GroupMessage:g100",
            message_str="监督 开始 @Bob 任务=做海报 待办=出图、排版 时长=2h 冷却=2h 抽取=2",
            messages=[Plain("监督 开始 "), At(name="Bob", qq="u200"), Plain(" 任务=做海报 待办=出图、排版 时长=2h 冷却=2h 抽取=2")],
            group_id="g100",
            private=False,
            admin=False,
        )
        non_admin_try = await collect_results(
            plugin.start_supervision(
                non_admin_group_event,
                "任务=做海报 待办=出图、排版 时长=2h 冷却=2h 抽取=2",
            )
        )
        assert non_admin_try and "管理员" in non_admin_try[0]
        passed.append("group_mention_requires_admin")

        admin_group_event = FakeEvent(
            sender_id="u999",
            sender_name="Boss",
            unified_msg_origin="aiocqhttp:GroupMessage:g100",
            message_str="监督 开始 @Bob 任务=做海报 待办=出图、排版 时长=2h 冷却=2h 抽取=2",
            messages=[Plain("监督 开始 "), At(name="Bob", qq="u200"), Plain(" 任务=做海报 待办=出图、排版 时长=2h 冷却=2h 抽取=2")],
            group_id="g100",
            private=False,
            admin=True,
        )
        admin_create = await collect_results(
            plugin.start_supervision(
                admin_group_event,
                "任务=做海报 待办=出图、排版 时长=2h 冷却=2h 抽取=2",
            )
        )
        assert admin_create and "Bob" in admin_create[0] and "做海报" in admin_create[0]
        passed.append("group_mention_admin_start")

        group_task_private_message = FakeEvent(
            sender_id="u200",
            sender_name="Bob",
            unified_msg_origin="aiocqhttp:FriendMessage:u200",
            message_str="private chat ping",
            messages=[Plain("private chat ping")],
            private=True,
        )
        await plugin.on_message(group_task_private_message)
        assert len(group_task_private_message.sent_messages) == 0
        passed.append("group_task_stays_in_group")

        target_message = FakeEvent(
            sender_id="u200",
            sender_name="Bob",
            unified_msg_origin="aiocqhttp:GroupMessage:g100",
            message_str="我来了",
            messages=[Plain("我来了")],
            group_id="g100",
            private=False,
        )
        await plugin.on_message(target_message)
        assert len(target_message.sent_messages) == 1
        assert target_message.call_llm is False
        assert target_message.stopped is True
        reminder_chain = target_message.sent_messages[0]
        reminder_plain = " ".join(plain_texts_of_chain(reminder_chain))
        assert "别磨蹭" in reminder_plain
        assert provider.calls, "provider.text_chat was not called"
        assert "毒舌女仆" in provider.calls[-1]["system_prompt"]
        passed.append("reminder_llm_path")

        target_message_second = FakeEvent(
            sender_id="u200",
            sender_name="Bob",
            unified_msg_origin="aiocqhttp:GroupMessage:g100",
            message_str="我又来了",
            messages=[Plain("我又来了")],
            group_id="g100",
            private=False,
        )
        await plugin.on_message(target_message_second)
        assert len(target_message_second.sent_messages) == 0
        passed.append("reminder_cooldown")

        admin_complete = FakeEvent(
            sender_id="u999",
            sender_name="Boss",
            unified_msg_origin="aiocqhttp:GroupMessage:g100",
            message_str="监督 完成 @Bob",
            messages=[At(name="Bob", qq="u200"), Plain(" 监督 完成")],
            group_id="g100",
            private=False,
            admin=True,
        )
        complete_group = await collect_results(plugin.complete_supervision(admin_complete))
        assert complete_group and "做海报" in complete_group[0]
        passed.append("group_complete")

        now_hhmm = plugin._now().strftime("%H:%M")
        broadcast_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str=f"更新内容 设置 时间={now_hhmm} 内容=今天更新了第一章",
            messages=[Plain(f"更新内容 设置 时间={now_hhmm} 内容=今天更新了第一章")],
            private=True,
        )
        update_set = await collect_results(plugin.update_set(broadcast_event, f"时间={now_hhmm}"))
        assert update_set and now_hhmm in update_set[0]
        passed.append("update_set")

        preview_event = FakeEvent(
            sender_id="u100",
            sender_name="Alice",
            unified_msg_origin="aiocqhttp:FriendMessage:u100",
            message_str=f"内容预告 设置 时间={now_hhmm} 内容=明天预告第二章",
            messages=[Plain(f"内容预告 设置 时间={now_hhmm} 内容=明天预告第二章")],
            private=True,
        )
        preview_set = await collect_results(
            plugin.preview_set(preview_event, f"时间={now_hhmm}")
        )
        assert preview_set and now_hhmm in preview_set[0]
        passed.append("preview_set")
        assert any(
            item.get("__template_key") == "update_broadcast"
            and item.get("time_hhmm") == now_hhmm
            and item.get("session_id") == "aiocqhttp:FriendMessage:u100"
            for item in plugin.config.get("broadcast_settings", [])
        )
        assert any(
            item.get("__template_key") == "preview_broadcast"
            and item.get("time_hhmm") == now_hhmm
            and item.get("session_id") == "aiocqhttp:FriendMessage:u100"
            for item in plugin.config.get("broadcast_settings", [])
        )
        passed.append("broadcast_command_sync")

        update_now = await collect_results(plugin.update_send_now(broadcast_event, ""))
        assert update_now and "已发送" in update_now[0]
        assert broadcast_event.sent_messages, "update_send_now did not send message"
        passed.append("update_send_now")

        plugin.config["broadcast_settings"] = [
            {
                "__template_key": "update_broadcast",
                "enabled": True,
                "session_id": "aiocqhttp:FriendMessage:u500",
                "session_label": "u500",
                "time_hhmm": now_hhmm,
                "content": "设置页更新内容",
            },
            {
                "__template_key": "preview_broadcast",
                "enabled": False,
                "session_id": "aiocqhttp:GroupMessage:g500",
                "session_label": "g500",
                "time_hhmm": now_hhmm,
                "content": "设置页内容预告",
            },
        ]
        await plugin._sync_broadcast_jobs_from_config(force=True)
        update_status_from_settings = await plugin._get_broadcast_status_text("update")
        preview_status_from_settings = await plugin._get_broadcast_status_text("preview")
        assert "u500" in update_status_from_settings and now_hhmm in update_status_from_settings
        assert "g500" in preview_status_from_settings and "关闭" in preview_status_from_settings
        passed.append("broadcast_settings_import")

        await plugin._run_due_broadcasts()
        assert len(context.sent_messages) == 1
        sent_bodies = [" ".join(item["plain_texts"]) for item in context.sent_messages]
        assert any("[更新内容]" in body for body in sent_bodies)
        assert not any("[内容预告]" in body for body in sent_bodies)
        assert any("设置页更新内容" in body for body in sent_bodies)
        passed.append("scheduled_broadcasts")

        plugin.config["broadcast_settings"] = []
        await plugin._sync_broadcast_jobs_from_config(force=True)
        cleared_update_status = await plugin._get_broadcast_status_text("update")
        cleared_preview_status = await plugin._get_broadcast_status_text("preview")
        assert "还没有设置" in cleared_update_status
        assert "还没有设置" in cleared_preview_status
        passed.append("broadcast_settings_clear")

        return passed
    finally:
        plugin_main.StarTools.get_data_dir = original_get_data_dir
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main() -> int:
    try:
        passed = await run_smoke_test()
    except AssertionError as exc:
        print(f"SMOKE TEST FAILED: {exc}")
        traceback.print_exc()
        return 1
    except Exception as exc:
        print(f"SMOKE TEST ERROR: {exc}")
        traceback.print_exc()
        return 1

    print("SMOKE TEST PASSED")
    for item in passed:
        print(f"PASS {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
