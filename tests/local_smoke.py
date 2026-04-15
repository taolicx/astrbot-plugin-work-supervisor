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

        self_status = await collect_results(plugin.status_supervision(self_event))
        assert self_status and "写第一章" in self_status[0] and "active" in self_status[0]
        passed.append("self_status")

        self_done = await collect_results(plugin.complete_supervision(self_event))
        assert self_done and "写第一章" in self_done[0]
        assert not any(
            item.get("task_title") == "写第一章"
            for item in plugin.config.get("settings_tasks", [])
        )
        passed.append("self_complete")

        no_task = await collect_results(plugin.status_supervision(self_event))
        assert no_task and "没有" in no_task[0]
        passed.append("self_status_after_complete")

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
                "duration": "2h",
                "cooldown": "30m",
                "todo_pick_count": 2,
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
        assert any(
            item.get("task_title") == "设置页任务"
            and item.get("session_id") == "aiocqhttp:FriendMessage:u300"
            and item.get("task_key")
            for item in plugin.config.get("settings_tasks", [])
        )
        passed.append("settings_task_import")

        config_done = await collect_results(plugin.complete_supervision(config_event))
        assert config_done and "设置页任务" in config_done[0]
        assert not any(
            item.get("task_title") == "设置页任务"
            for item in plugin.config.get("settings_tasks", [])
        )
        passed.append("settings_task_complete_sync")

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

        update_now = await collect_results(plugin.update_send_now(broadcast_event, ""))
        assert update_now and "已发送" in update_now[0]
        assert broadcast_event.sent_messages, "update_send_now did not send message"
        passed.append("update_send_now")

        await plugin._run_due_broadcasts()
        assert len(context.sent_messages) == 2
        sent_bodies = [" ".join(item["plain_texts"]) for item in context.sent_messages]
        assert any("[更新内容]" in body for body in sent_bodies)
        assert any("[内容预告]" in body for body in sent_bodies)
        assert any("今天更新了第一章" in body for body in sent_bodies)
        assert any("明天预告第二章" in body for body in sent_bodies)
        passed.append("scheduled_broadcasts")

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
