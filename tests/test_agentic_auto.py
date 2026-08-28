import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("AGENTHUB_STORAGE_DIR", tempfile.mkdtemp(prefix="agenthub-auto-tests-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
import orchestrator  # noqa: E402
from tools.mcp.registry import ALL_TOOLS  # noqa: E402

app.config.AUTO_PLANNER_USE_LLM = False


class AgenticAutoPlannerTests(unittest.TestCase):
    def context(self, **kwargs):
        defaults = {
            "has_selected_database": True,
            "has_rag_documents": False,
            "has_sql_context": False,
            "has_last_answer": False,
        }
        defaults.update(kwargs)
        return orchestrator.PlanningContext(**defaults)

    def test_sql_only_turn(self):
        plan = orchestrator.plan_turn("How many employees work in Engineering?", self.context())
        self.assertEqual([s.capability for s in plan.steps], ["sql_agent"])

    def test_sql_followup_then_gmail_switches_capability(self):
        plan = orchestrator.plan_turn("email me that result", self.context(has_last_answer=True, has_sql_context=True))
        self.assertEqual([(s.capability, s.action) for s in plan.steps], [("gmail", "send_email")])

    def test_sql_then_email_combined_goal(self):
        plan = orchestrator.plan_turn(
            "Find the highest paid employee and email me their name.",
            self.context(),
        )
        self.assertEqual([s.capability for s in plan.steps], ["sql_agent", "gmail"])

    def test_calendar_only_turn(self):
        plan = orchestrator.plan_turn("What meetings do I have tomorrow?", self.context())
        self.assertEqual([(s.capability, s.action) for s in plan.steps], [("google_calendar", "check_schedule")])

    def test_calendar_then_gmail_multi_tool_turn(self):
        plan = orchestrator.plan_turn("Move my 3 PM meeting and email everyone", self.context())
        self.assertEqual([s.capability for s in plan.steps], ["google_calendar", "gmail"])

    def test_leave_request_does_not_inherit_database_routing(self):
        plan = orchestrator.plan_turn("request leave on Aug 15", self.context(has_sql_context=True))
        self.assertEqual(plan.steps[0].capability, "date_time")
        self.assertNotIn("sql_agent", [s.capability for s in plan.steps])
        # Holiday status is now checked for real in the Date/Time tool's own
        # observation (see app.py: _holiday_status_line), so the only thing
        # still missing at the planning stage is the recipient.
        self.assertIn("recipient", plan.missing_info)

    def test_general_chat_uses_no_tool(self):
        plan = orchestrator.plan_turn("Tell me a concise joke", self.context(has_selected_database=False))
        self.assertEqual([(s.capability, s.action) for s in plan.steps], [("general_chat", "respond")])

    def test_rag_then_gmail_when_documents_are_available(self):
        plan = orchestrator.plan_turn(
            "find sick leave policy in my uploaded docs and email it to me",
            self.context(has_rag_documents=True),
        )
        self.assertEqual([s.capability for s in plan.steps], ["rag_agent", "gmail"])

    def test_github_is_not_registered_as_project_tool(self):
        self.assertNotIn("github", [tool.key for tool in ALL_TOOLS])
        self.assertNotIn("github", app.oauth.integration_status())


class AgenticAutoNoModeSwitchTests(unittest.TestCase):
    def context(self, **kwargs):
        defaults = {
            "has_selected_database": True,
            "has_rag_documents": False,
            "has_sql_context": False,
            "has_last_answer": False,
        }
        defaults.update(kwargs)
        return orchestrator.PlanningContext(**defaults)

    def test_who_are_you_stays_general_with_db_selected(self):
        plan = orchestrator.plan_turn("Who are you?", self.context())
        self.assertEqual([s.capability for s in plan.steps], ["general_chat"])

    def test_explain_recursion_stays_general(self):
        plan = orchestrator.plan_turn("Explain recursion.", self.context())
        self.assertEqual([s.capability for s in plan.steps], ["general_chat"])

    def test_sql_without_database_asks_for_db(self):
        plan = orchestrator.plan_turn("How many employees are there?", self.context(has_selected_database=False))
        self.assertEqual(plan.intent, "sql_needs_database")
        self.assertIn("database", plan.missing_info.lower())
        self.assertEqual(plan.steps, [])

    def test_document_question_without_docs_asks_to_attach(self):
        plan = orchestrator.plan_turn(
            "What does my employee handbook say about sick leave?",
            self.context(has_rag_documents=False),
        )
        self.assertEqual(plan.intent, "rag_needs_document")
        self.assertIn("attach", plan.missing_info.lower())
        self.assertNotIn("mode", plan.missing_info.lower())

    def test_attached_file_heading_routes_to_rag(self):
        plan = orchestrator.plan_turn(
            "what is this, give me the heading for this file",
            self.context(has_rag_documents=True),
        )
        self.assertEqual([s.capability for s in plan.steps], ["rag_agent"])

    def test_attached_file_does_not_force_rag_for_identity(self):
        plan = orchestrator.plan_turn("Who are you?", self.context(has_rag_documents=True))
        self.assertEqual([s.capability for s in plan.steps], ["general_chat"])

    def test_attached_file_does_not_hijack_self_capability_questions(self):
        for message in (
            "What do you do?",
            "What is it that you do?",
            "what is it that u do?",
            "how do u help me?",
            "How can you help me?",
        ):
            with self.subTest(message=message):
                plan = orchestrator.plan_turn(message, self.context(has_rag_documents=True))
                self.assertEqual([s.capability for s in plan.steps], ["general_chat"])

    def test_llm_rag_step_is_rejected_for_self_question(self):
        raw_plan = orchestrator.LLMPlannerOutput(
            intent="rag",
            goal="Explain the assistant",
            tools_needed=True,
            steps=[
                orchestrator.LLMPlanStep(
                    capability="rag_agent",
                    action="query",
                    reason="Incorrectly interpreted 'it' as the attachment.",
                )
            ],
        )
        for message in ("what is it that u do?", "how do u help me?"):
            with self.subTest(message=message):
                plan = orchestrator._validate_llm_plan(  # noqa: SLF001 - planner boundary regression
                    raw_plan,
                    message,
                    self.context(has_rag_documents=True),
                )
                self.assertEqual([s.capability for s in plan.steps], ["general_chat"])

    def test_llm_rag_step_is_rejected_for_unrelated_normal_question(self):
        raw_plan = orchestrator.LLMPlannerOutput(
            intent="rag",
            goal="Answer a normal question",
            tools_needed=True,
            steps=[
                orchestrator.LLMPlanStep(
                    capability="rag_agent",
                    action="query",
                    reason="Incorrectly selected because attachments exist.",
                )
            ],
        )
        plan = orchestrator._validate_llm_plan(  # noqa: SLF001 - planner boundary regression
            raw_plan,
            "How can I improve my focus?",
            self.context(has_selected_database=False, has_rag_documents=True),
        )
        self.assertEqual([(s.capability, s.action) for s in plan.steps], [("general_chat", "respond")])

    def test_attached_file_deixis_still_routes_to_rag(self):
        plan = orchestrator.plan_turn(
            "What does it say?",
            self.context(has_rag_documents=True),
        )
        self.assertEqual([(s.capability, s.action) for s in plan.steps], [("rag_agent", "query")])

    def test_explicit_document_help_question_still_routes_to_rag(self):
        plan = orchestrator.plan_turn(
            "How can you help me with this PDF?",
            self.context(has_rag_documents=True),
        )
        self.assertEqual([(s.capability, s.action) for s in plan.steps], [("rag_agent", "query")])

    def test_relative_dates_route_to_date_time(self):
        for message in ("What date is today?", "What date is tomorrow?", "What date is tommorow?"):
            with self.subTest(message=message):
                plan = orchestrator.plan_turn(message, self.context(has_selected_database=False))
                self.assertEqual([s.capability for s in plan.steps], ["date_time"])

    def test_compact_month_date_routes_to_date_time(self):
        plan = orchestrator.plan_turn("What day is aug16th?", self.context(has_selected_database=False))
        self.assertEqual([s.capability for s in plan.steps], ["date_time"])

    def test_attached_file_weather_not_rag(self):
        plan = orchestrator.plan_turn("What's the weather in Tampa?", self.context(has_rag_documents=True))
        self.assertEqual([s.capability for s in plan.steps], ["weather"])

    def test_leave_request_not_sql(self):
        plan = orchestrator.plan_turn("Request leave on Aug 15.", self.context())
        self.assertNotIn("sql_agent", [s.capability for s in plan.steps])

    def test_leave_request_missing_date_asks_only_for_date(self):
        plan = orchestrator.plan_turn("Can you request leave to manager@example.com?", self.context())
        self.assertEqual(plan.intent, "leave_needs_date")
        self.assertEqual(plan.steps, [])
        self.assertIn("what date", plan.missing_info.lower())

    def test_llm_planner_none_falls_back_to_rules(self):
        with patch.object(orchestrator, "_plan_turn_with_llm", return_value=None):
            plan = orchestrator.plan_turn(
                "How many employees work in Engineering?",
                self.context(use_llm_planner=True),
            )
        self.assertEqual([s.capability for s in plan.steps], ["sql_agent"])

    def test_llm_planner_cannot_downgrade_leave_request_to_general_chat(self):
        bad_llm_plan = orchestrator.AutoPlan(
            goal="request leave on Aug 15",
            intent="general_conversation",
            steps=[orchestrator.PlanStep("general_chat", "respond", "Bad downgrade.")],
        )
        with patch.object(orchestrator, "_plan_turn_with_llm", return_value=bad_llm_plan):
            plan = orchestrator.plan_turn(
                "request leave on Aug 15",
                self.context(use_llm_planner=True),
            )
        self.assertNotEqual([s.capability for s in plan.steps], ["general_chat"])
        self.assertEqual(plan.steps[0].capability, "date_time")

    def test_llm_planner_cannot_downgrade_email_missing_info_to_general_chat(self):
        bad_llm_plan = orchestrator.AutoPlan(
            goal="send an email saying hello",
            intent="general_conversation",
            steps=[orchestrator.PlanStep("general_chat", "respond", "Bad downgrade.")],
        )
        with patch.object(orchestrator, "_plan_turn_with_llm", return_value=bad_llm_plan):
            plan = orchestrator.plan_turn(
                "send an email saying hello",
                self.context(use_llm_planner=True),
            )
        self.assertEqual(plan.steps, [])
        self.assertEqual(plan.intent, "email_needs_recipient")

    def test_send_an_email_alone_asks_for_recipient_and_body(self):
        plan = orchestrator.plan_turn("send an email", self.context(has_selected_database=False))
        self.assertEqual(plan.steps, [])
        self.assertIn("email_recipient", plan.missing_fields)
        self.assertIn("email_body", plan.missing_fields)
        self.assertIn("who should i send it to", plan.missing_info.lower())
        self.assertIn("what should it say", plan.missing_info.lower())


class DateTimeParsingTests(unittest.TestCase):
    def test_month_day_ordinal_resolves_against_current_year_when_future(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 8, 13, 9, 0, 0)

        with patch.object(app, "datetime", FixedDateTime):
            success, reply = app.execute_date_time("October 2nd")
        self.assertTrue(success)
        self.assertIn("2026-10-02", reply)

    def test_next_friday_resolves_from_current_date(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 8, 13, 9, 0, 0)

        with patch.object(app, "datetime", FixedDateTime):
            success, reply = app.execute_date_time("next Friday")
        self.assertTrue(success)
        self.assertIn("2026-08-14", reply)

    def test_compact_month_day_and_range_resolve(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 8, 13, 9, 0, 0)

        with patch.object(app, "datetime", FixedDateTime):
            success, reply = app.execute_date_time("aug16th-17th")
        self.assertTrue(success)
        self.assertIn("2026-08-16", reply)
        self.assertIn("2026-08-17", reply)

    def test_compact_ordinal_day_first_resolves(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 8, 13, 9, 0, 0)

        with patch.object(app, "datetime", FixedDateTime):
            success, reply = app.execute_date_time("16thAug")
        self.assertTrue(success)
        self.assertIn("2026-08-16", reply)

    def test_invalid_date_reply_gives_usable_examples(self):
        success, reply = app.execute_date_time("August 40")
        self.assertFalse(success)
        self.assertIn("August 16", reply)
        self.assertIn("tomorrow", reply)

    def test_month_pattern_does_not_partially_match_long_numbers(self):
        self.assertIsNone(orchestrator.MONTH_FIRST_DATE_RE.search("aug2026"))


class AgenticAutoHandlerTests(unittest.TestCase):
    def setUp(self):
        app._PENDING_MCP_ACTIONS.clear()
        app._THREAD_AUTO_CONTEXT.clear()
        app._THREAD_PENDING_GOAL.clear()
        self.thread_id = "test-thread"
        self.db_name = app.default_db_choice()

    def chat(self, message, history=None):
        return app.handle_message(
            app.ChatRequest(
                message=message,
                thread_id=self.thread_id,
                mode="Auto",
                db_name=self.db_name,
                history=history or [],
            )
        )

    def test_sql_only_turn_uses_real_sql_observation_shape(self):
        fake_sql = {
            "final_answer": "Engineering has 4 employees.",
            "sql_query": "SELECT COUNT(*) FROM employees;",
            "sql_result": {"columns": ["count"], "rows": [(4,)]},
            "retry_history": [],
            "validation_error": "",
            "retry_count": 0,
            "execution_success": True,
        }
        with patch.object(app.compiled_app, "invoke", return_value=fake_sql):
            response = self.chat("How many employees work in Engineering?")
        self.assertEqual(response.mode_used, "Agentic Auto -> SQL")
        self.assertEqual(response.result["rows"], [[4]])
        self.assertTrue(response.sql_valid)

    def test_sql_then_gmail_followup_requires_approval_before_send(self):
        app._THREAD_AUTO_CONTEXT[self.thread_id] = {"last_answer": "Engineering has 4 employees."}
        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ):
            response = self.chat("email me that result")
        self.assertIn("Reply **yes** to approve", response.reply)
        self.assertIn(self.thread_id, app._PENDING_MCP_ACTIONS)

    def test_gmail_approval_uses_mocked_api_confirmation(self):
        app._THREAD_AUTO_CONTEXT[self.thread_id] = {"last_answer": "Engineering has 4 employees."}
        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ):
            self.chat("email me that result")

        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ), patch.object(app.oauth, "google_access_token", return_value="token"), patch.object(
            app.oauth, "post_json", return_value={"id": "mock-message-id"}
        ) as post_json:
            response = self.chat("yes")
        self.assertIn("mock-message-id", response.reply)
        post_json.assert_called_once()

    def test_repeated_gmail_approval_does_not_duplicate_send(self):
        app._THREAD_AUTO_CONTEXT[self.thread_id] = {"last_answer": "Engineering has 4 employees."}
        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ):
            self.chat("email me that result")

        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ), patch.object(app.oauth, "google_access_token", return_value="token"), patch.object(
            app.oauth, "post_json", return_value={"id": "mock-message-id"}
        ) as post_json, patch.object(app.general_chat, "respond", return_value="Yes noted."):
            first = self.chat("yes")
            second = self.chat("yes")
        self.assertIn("mock-message-id", first.reply)
        self.assertEqual(second.mode_used, "Agentic Auto")
        self.assertIn("nothing to approve", second.reply.lower())
        post_json.assert_called_once()

    def test_cancel_pending_gmail_approval_does_not_send(self):
        app._THREAD_AUTO_CONTEXT[self.thread_id] = {"last_answer": "Engineering has 4 employees."}
        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ):
            self.chat("email me that result")

        with patch.object(app.oauth, "post_json") as post_json:
            response = self.chat("cancel")
        self.assertIn("cancelled", response.reply.lower())
        self.assertNotIn(self.thread_id, app._PENDING_MCP_ACTIONS)
        post_json.assert_not_called()

    def test_gmail_tool_failure_does_not_claim_success(self):
        app._THREAD_AUTO_CONTEXT[self.thread_id] = {"last_answer": "Engineering has 4 employees."}
        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ):
            self.chat("email me that result")

        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ), patch.object(app.oauth, "google_access_token", return_value="token"), patch.object(
            app.oauth, "post_json", side_effect=RuntimeError("gmail 500")
        ):
            response = self.chat("yes")
        self.assertIn("failed", response.reply.lower())
        self.assertIn("gmail 500", response.reply)
        self.assertNotIn(self.thread_id, app._PENDING_MCP_ACTIONS)

    def test_combined_sql_then_gmail_previews_real_sql_result(self):
        fake_sql = {
            "final_answer": "Asha has the highest salary at 140000.",
            "sql_query": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1;",
            "sql_result": {"columns": ["name", "salary"], "rows": [("Asha", 140000)]},
            "retry_history": [],
            "validation_error": "",
            "retry_count": 0,
            "execution_success": True,
        }
        with patch.object(app.compiled_app, "invoke", return_value=fake_sql), patch.object(
            app.oauth, "is_tool_connected", return_value=True
        ), patch.object(app.oauth, "google_profile", return_value={"email": "me@example.com"}):
            response = self.chat("Find the highest paid employee and email me the result")
        self.assertEqual(response.mode_used, "Agentic Auto -> SQL -> Gmail")
        self.assertIn("Asha", response.reply)
        self.assertIn("Reply **yes** to approve", response.reply)
        self.assertEqual(app._PENDING_MCP_ACTIONS[self.thread_id]["arguments"]["to_email"], "me@example.com")

    def test_unrelated_yes_without_pending_never_reaches_the_chat_model(self):
        # A bare approval must not be answered by the chat model, which would
        # happily invent "your email has been sent" for an action that never ran.
        with patch.object(app.general_chat, "respond", return_value="Sure, I've sent it.") as respond:
            response = self.chat("yes")
        respond.assert_not_called()
        self.assertEqual(response.mode_used, "Agentic Auto")
        self.assertIn("nothing to approve", response.reply.lower())
        self.assertNotIn("sent", response.reply.lower().replace("nothing has been sent", ""))

    def test_email_with_recipient_missing_body_then_resumes_to_preview(self):
        with patch.object(app.config, "AUTO_PLANNER_USE_LLM", False), patch.object(
            app.oauth, "is_tool_connected", return_value=True
        ), patch.object(app.oauth, "google_profile", return_value={"email": "me@example.com"}):
            first = self.chat("send an email to bob@example.com")
            second = self.chat("Please review the proposal.")
        self.assertIn("what should the email say", first.reply.lower())
        self.assertIn("Reply **yes** to approve", second.reply)
        self.assertIn("Please review the proposal", second.reply)

    def test_gmail_draft_creation_uses_real_pending_write_path(self):
        with patch.object(app.config, "AUTO_PLANNER_USE_LLM", False), patch.object(
            app.oauth, "is_tool_connected", return_value=True
        ), patch.object(app.oauth, "google_profile", return_value={"email": "me@example.com"}):
            preview = self.chat("draft an email to bob@example.com saying please review the proposal")
        self.assertIn("Gmail draft ready", preview.reply)
        self.assertIn("Reply **yes** to approve", preview.reply)
        self.assertEqual(app._PENDING_MCP_ACTIONS[self.thread_id]["action_name"], "draft_reply")

        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ), patch.object(app.oauth, "google_access_token", return_value="token"), patch.object(
            app.oauth, "post_json", return_value={"id": "draft-123", "message": {"id": "msg-456"}}
        ) as post_json:
            response = self.chat("yes")
        self.assertIn("draft-123", response.reply)
        post_json.assert_called_once()

    def test_send_email_missing_details_then_resume_preview_and_mocked_send(self):
        with patch.object(app.config, "AUTO_PLANNER_USE_LLM", False), patch.object(
            app.oauth, "is_tool_connected", return_value=True
        ), patch.object(app.oauth, "google_profile", return_value={"email": "me@example.com"}):
            first = self.chat("send an email")
            self.assertIn("who should i send it to", first.reply.lower())
            self.assertIn("what should it say", first.reply.lower())
            self.assertNotIn(self.thread_id, app._PENDING_MCP_ACTIONS)
            self.assertIn(self.thread_id, app._THREAD_PENDING_GOAL)

            second = self.chat(
                "to manager@example.com saying I need leave on Aug 18 because I am sick"
            )
        self.assertIn("Email draft ready", second.reply)
        self.assertIn("manager@example.com", second.reply)
        self.assertIn("Leave request for August 18", second.reply)
        self.assertIn("I need leave on Aug 18 because I am sick", second.reply)
        self.assertNotIn("Date/Time", second.mode_used)
        self.assertIn("Reply **yes** to approve", second.reply)
        pending = app._PENDING_MCP_ACTIONS[self.thread_id]
        self.assertEqual(pending["arguments"]["to_email"], "manager@example.com")
        self.assertEqual(pending["status"], "awaiting_approval")

        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ), patch.object(app.oauth, "google_access_token", return_value="token"), patch.object(
            app.oauth, "post_json", return_value={"id": "gmail-msg-123"}
        ) as post_json, patch.object(app.general_chat, "respond", return_value="plain yes"):
            third = self.chat("yes")
            fourth = self.chat("yes")
        self.assertIn("gmail-msg-123", third.reply)
        self.assertIn("manager@example.com", third.reply)
        post_json.assert_called_once()
        self.assertNotIn(self.thread_id, app._PENDING_MCP_ACTIONS)
        self.assertEqual(fourth.mode_used, "Agentic Auto")

    def test_send_email_routes_to_gmail_not_general_chat(self):
        with patch.object(app.config, "AUTO_PLANNER_USE_LLM", False), patch.object(
            app.general_chat, "respond", return_value="should not be used"
        ) as respond:
            response = self.chat("send an email")
        respond.assert_not_called()
        self.assertNotIn("General Chat", response.mode_used)
        self.assertIn("who should i send", response.reply.lower())

    def test_calendar_only_uses_real_read_when_connected(self):
        calendar_payload = {
            "items": [
                {
                    "summary": "Design review",
                    "start": {"dateTime": "2026-08-14T10:00:00Z"},
                    "end": {"dateTime": "2026-08-14T10:30:00Z"},
                }
            ]
        }
        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_access_token", return_value="token"
        ), patch.object(app.oauth, "get_json", return_value=calendar_payload) as get_json:
            response = self.chat("What meetings do I have tomorrow?")
        self.assertEqual(response.mode_used, "Agentic Auto -> Google Calendar")
        self.assertIn("Design review", response.reply)
        get_json.assert_called_once()

    def test_calendar_then_gmail_stops_when_calendar_has_no_real_result(self):
        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_access_token", return_value="token"
        ), patch.object(app.oauth, "get_json", return_value={"items": []}):
            response = self.chat("Move my 3 PM meeting and email everyone")
        self.assertEqual(response.mode_used, "Agentic Auto -> Google Calendar")
        self.assertIn("No events found", response.reply)
        self.assertNotIn(self.thread_id, app._PENDING_MCP_ACTIONS)

    def test_clear_non_database_request_does_not_call_sql(self):
        with patch.object(app.compiled_app, "invoke") as invoke:
            response = self.chat("request leave on Aug 15")
        invoke.assert_not_called()
        self.assertIn("Date/Time", response.mode_used)
        # Holiday status is now checked for real via the offline `holidays`
        # calendar (see app.py: _holiday_status_line) instead of the old
        # placeholder that always said it wasn't verified.
        self.assertIn("Holiday check", response.reply)

    def test_missing_required_email_info_does_not_queue_write(self):
        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ):
            response = self.chat("send an email saying hello")
        self.assertIn("who should i send", response.reply.lower())
        self.assertNotIn(self.thread_id, app._PENDING_MCP_ACTIONS)

    def test_pending_database_goal_resumes_after_named_db(self):
        first = app.handle_message(
            app.ChatRequest(
                message="How many employees are there?",
                thread_id=self.thread_id,
                mode="Auto",
                db_name=None,
                history=[],
            )
        )
        self.assertIn("which database", first.reply.lower())
        self.assertIn(self.thread_id, app._THREAD_PENDING_GOAL)

        fake_sql = {
            "final_answer": "There are 12 employees.",
            "sql_query": "SELECT COUNT(*) FROM employees;",
            "sql_result": {"columns": ["count"], "rows": [(12,)]},
            "retry_history": [],
            "validation_error": "",
            "retry_count": 0,
            "execution_success": True,
        }
        with patch.object(app.compiled_app, "invoke", return_value=fake_sql) as invoke:
            second = app.handle_message(
                app.ChatRequest(
                    message=self.db_name or "sample_company.db",
                    thread_id=self.thread_id,
                    mode="Auto",
                    db_name=None,
                    history=[],
                )
            )
        invoke.assert_called_once()
        self.assertEqual(second.mode_used, "Agentic Auto -> SQL")
        self.assertIn("12", second.reply)

    def test_auto_general_chat_uses_auto_prompt(self):
        with patch.object(app.general_chat, "respond", return_value="I'm AgentHub Studio.") as respond:
            response = self.chat("Who are you?")
        respond.assert_called_once()
        self.assertTrue(respond.call_args.kwargs.get("auto_mode"))
        self.assertEqual(response.mode_used, "Agentic Auto -> General Chat")
        self.assertNotIn("switch", response.reply.lower())

    def test_general_chat_path(self):
        with patch.object(app.general_chat, "respond", return_value="Hello from general chat."):
            response = self.chat("Tell me a concise joke", history=[])
        self.assertEqual(response.mode_used, "Agentic Auto -> General Chat")
        self.assertEqual(response.reply, "Hello from general chat.")

    def test_leave_request_with_recipient_and_natural_date_previews_gmail_only(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 8, 13, 9, 0, 0)

        with patch.object(app, "datetime", FixedDateTime), patch.object(
            app.oauth, "is_tool_connected", return_value=True
        ), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ):
            response = self.chat("can you request a leave to my manager (manager@example.com) on october 2nd")
        self.assertEqual(response.mode_used, "Agentic Auto -> Date/Time -> Gmail")
        self.assertIn("Reply **yes** to approve", response.reply)
        self.assertIn("To: manager@example.com", response.reply)
        self.assertIn("Subject: Leave request for October 2, 2026", response.reply)
        self.assertIn("October 2, 2026", response.reply)
        self.assertIn(self.thread_id, app._PENDING_MCP_ACTIONS)

    def test_leave_request_missing_recipient_then_resumes_to_preview(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 8, 13, 9, 0, 0)

        with patch.object(app, "datetime", FixedDateTime):
            first = self.chat("can you request a leave on october 2nd")
        self.assertIn("recipient", first.reply.lower())
        self.assertIn(self.thread_id, app._THREAD_PENDING_GOAL)

        with patch.object(app, "datetime", FixedDateTime), patch.object(
            app.oauth, "is_tool_connected", return_value=True
        ), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ):
            second = self.chat("manager@example.com")
        self.assertIn("Reply **yes** to approve", second.reply)
        self.assertIn("To: manager@example.com", second.reply)

    def test_leave_request_missing_date_then_resumes_to_preview(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 8, 13, 9, 0, 0)

        first = self.chat("can you request a leave to manager@example.com")
        self.assertIn("what date", first.reply.lower())
        self.assertIn(self.thread_id, app._THREAD_PENDING_GOAL)

        with patch.object(app, "datetime", FixedDateTime), patch.object(
            app.oauth, "is_tool_connected", return_value=True
        ), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ):
            second = self.chat("october 2nd")
        self.assertIn("Reply **yes** to approve", second.reply)
        self.assertIn("To: manager@example.com", second.reply)
        self.assertIn("October 2, 2026", second.reply)

    def test_attached_doc_direct_question_uses_rag_sources(self):
        rag_payload = {
            "answer": "This file is about AgentHub Auto routing.",
            "sources": [{"file_name": "auto-routing.txt", "chunk_index": 0, "score": 0.91}],
            "retrieved_contexts": [{"text": "AgentHub Auto routing", "metadata": {"file_name": "auto-routing.txt"}}],
            "debug": {"intent": "qa", "retrieval_count": 1, "documents_searched": 1},
        }
        with patch.object(app, "_rag_scope_readiness", return_value="ready"), patch.object(
            app, "execute_rag_query", return_value=rag_payload
        ) as rag_query:
            response = app.handle_message(
                app.ChatRequest(
                    message="what is this file about?",
                    thread_id=self.thread_id,
                    mode="Auto",
                    db_name=self.db_name,
                    document_ids=["doc-test"],
                    history=[],
                )
            )
        rag_query.assert_called_once()
        self.assertEqual(response.mode_used, "Agentic Auto -> RAG")
        self.assertEqual(response.sources, rag_payload["sources"])

    def test_attached_doc_unrelated_general_question_does_not_use_rag(self):
        with patch.object(app, "execute_rag_query") as rag_query, patch.object(
            app.general_chat, "respond", return_value="I'm AgentHub Studio."
        ):
            response = app.handle_message(
                app.ChatRequest(
                    message="who are you?",
                    thread_id=self.thread_id,
                    mode="Auto",
                    db_name=self.db_name,
                    document_ids=["doc-test"],
                    history=[],
                )
            )
        rag_query.assert_not_called()
        self.assertEqual(response.mode_used, "Agentic Auto -> General Chat")

    def test_github_is_not_an_auto_tool_route(self):
        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.general_chat, "respond", return_value="GitHub is not part of this workspace."
        ):
            response = self.chat("Analyze this GitHub repo")
        self.assertEqual(response.mode_used, "Agentic Auto -> General Chat")
        self.assertNotIn("not implemented yet", response.reply)

    def test_calendar_write_missing_info_resumes_to_preview(self):
        first = self.chat("schedule a meeting")
        self.assertIn("date", first.reply.lower())
        self.assertIn("time", first.reply.lower())
        self.assertIn("attendee", first.reply.lower())

        second = self.chat("Friday")
        self.assertNotIn("date, time, attendee", second.reply.lower())
        self.assertIn("time", second.reply.lower())
        self.assertIn("attendee", second.reply.lower())

        with patch.object(app.oauth, "is_tool_connected", return_value=True):
            third = self.chat("at 3pm with bob@example.com")
        self.assertIn("Reply **yes** to approve", third.reply)
        self.assertIn("bob@example.com", third.reply)
        self.assertEqual(app._PENDING_MCP_ACTIONS[self.thread_id]["tool_key"], "google_calendar")

    def test_calendar_write_approval_uses_mocked_api_confirmation(self):
        with patch.object(app.oauth, "is_tool_connected", return_value=True):
            preview = self.chat("schedule a meeting Friday at 3pm with bob@example.com")
        self.assertIn("Reply **yes** to approve", preview.reply)

        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_access_token", return_value="token"
        ), patch.object(app.oauth, "post_json", return_value={"id": "event-123"}) as post_json:
            response = self.chat("approve")
        self.assertIn("event-123", response.reply)
        post_json.assert_called_once()


def _ask_user_plan(message, question):
    """Mirrors what orchestrator._validate_llm_plan builds when the live LLM
    planner answers with next_action='ask_user': a plan with no steps at all."""
    return orchestrator.AutoPlan(
        goal=message,
        intent="leave_request",
        steps=[],
        missing_info=question,
        completion_status="needs_user",
    )


class AgenticAutoLLMPlannerTests(unittest.TestCase):
    """The live default is AUTO_PLANNER_USE_LLM=true, but every other test runs
    with it off. These cover the path that actually shipped."""

    def setUp(self):
        app._PENDING_MCP_ACTIONS.clear()
        app._THREAD_AUTO_CONTEXT.clear()
        app._THREAD_PENDING_GOAL.clear()
        self.thread_id = "llm-planner-thread"
        self.db_name = app.default_db_choice()

    def chat(self, message):
        return app.handle_message(
            app.ChatRequest(
                message=message,
                thread_id=self.thread_id,
                mode="Auto",
                db_name=self.db_name,
                history=[],
            )
        )

    def context(self, **kwargs):
        defaults = {
            "has_selected_database": True,
            "has_rag_documents": False,
            "has_sql_context": False,
            "has_last_answer": False,
            "use_llm_planner": True,
        }
        defaults.update(kwargs)
        return orchestrator.PlanningContext(**defaults)

    def test_ask_user_plan_cannot_discard_a_gmail_step(self):
        with patch.object(
            orchestrator,
            "_plan_turn_with_llm",
            side_effect=lambda message, context: _ask_user_plan(message, "Approval to send the email is required."),
        ):
            plan = orchestrator.plan_turn("email me that result", self.context(has_last_answer=True))
        self.assertEqual([(s.capability, s.action) for s in plan.steps], [("gmail", "send_email")])

    def test_ask_user_plan_cannot_discard_the_leave_workflow(self):
        with patch.object(
            orchestrator,
            "_plan_turn_with_llm",
            side_effect=lambda message, context: _ask_user_plan(message, "Please provide the reason for your leave."),
        ):
            plan = orchestrator.plan_turn(
                "request a leave on october 2nd to manager@example.com",
                self.context(),
            )
        self.assertEqual([s.capability for s in plan.steps], ["date_time", "gmail"])

    def test_ask_user_plan_cannot_replace_the_rule_question(self):
        with patch.object(
            orchestrator,
            "_plan_turn_with_llm",
            side_effect=lambda message, context: _ask_user_plan(message, "How many days of leave do you need?"),
        ):
            plan = orchestrator.plan_turn("can you request leave", self.context())
        self.assertEqual(plan.intent, "leave_needs_date")
        self.assertEqual(plan.missing_fields, ["date"])

    def test_llm_operational_plan_is_still_honoured(self):
        llm_plan = orchestrator.AutoPlan(
            goal="check my calendar",
            intent="calendar",
            steps=[orchestrator.PlanStep("google_calendar", "check_schedule", "Read the schedule.")],
        )
        with patch.object(orchestrator, "_plan_turn_with_llm", return_value=llm_plan):
            plan = orchestrator.plan_turn("what meetings do I have tomorrow?", self.context())
        self.assertEqual([s.capability for s in plan.steps], ["google_calendar"])

    def test_leave_conversation_reaches_a_real_send_despite_constant_ask_user(self):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls):
                return cls(2026, 8, 13, 9, 0, 0)

        interrogate = lambda message, context: _ask_user_plan(  # noqa: E731
            message, "Please confirm the duration of the leave and any specific details."
        )
        with patch.object(app.config, "AUTO_PLANNER_USE_LLM", True), patch.object(
            orchestrator, "_plan_turn_with_llm", side_effect=interrogate
        ), patch.object(app, "datetime", FixedDateTime), patch.object(
            app.oauth, "is_tool_connected", return_value=True
        ), patch.object(app.oauth, "google_profile", return_value={"email": "me@example.com"}), patch.object(
            app.general_chat, "respond", return_value="should not be needed"
        ):
            first = self.chat("request for a leave tommorow")
            second = self.chat("madalaabhay1@gmail.com")

        self.assertIn("recipient", first.reply.lower())
        self.assertNotIn("duration", first.reply.lower())
        self.assertIn("Reply **yes** to approve", second.reply)
        self.assertIn("To: madalaabhay1@gmail.com", second.reply)
        self.assertIn("August 14, 2026", second.reply)
        self.assertEqual(app._PENDING_MCP_ACTIONS[self.thread_id]["tool_key"], "gmail")

        with patch.object(app.oauth, "is_tool_connected", return_value=True), patch.object(
            app.oauth, "google_profile", return_value={"email": "me@example.com"}
        ), patch.object(app.oauth, "google_access_token", return_value="token"), patch.object(
            app.oauth, "post_json", return_value={"id": "gmail-real-send"}
        ) as post_json:
            third = self.chat("yes")
        self.assertIn("gmail-real-send", third.reply)
        post_json.assert_called_once()

    def test_free_form_question_keeps_the_goal_and_collected_slots(self):
        app._THREAD_PENDING_GOAL[self.thread_id] = {
            "conversation_id": self.thread_id,
            "status": "waiting_for_fields",
            "missing": "email_recipient",
            "missing_fields": ["email_recipient"],
            "field_values": {"email_recipient": "manager@example.com"},
            "observations": [],
            "pending_write": None,
            "goal": "request a leave tomorrow",
            "message": "request a leave tomorrow",
            "intent": "leave_request",
        }
        plan = orchestrator.AutoPlan(goal="Request sick leave.", intent="leave_request", steps=[])
        plan.missing_info = "Approval to send the email is required."

        app.remember_pending_auto_goal(self.thread_id, "request a leave tomorrow", plan)

        pending = app._THREAD_PENDING_GOAL[self.thread_id]
        self.assertEqual(pending["goal"], "request a leave tomorrow")
        self.assertEqual(pending["field_values"]["email_recipient"], "manager@example.com")

    def test_reply_that_fills_no_slot_is_still_passed_to_the_planner(self):
        app._THREAD_PENDING_GOAL[self.thread_id] = {
            "conversation_id": self.thread_id,
            "status": "waiting_for_fields",
            "missing": None,
            "missing_fields": [],
            "field_values": {},
            "notes": [],
            "observations": [],
            "pending_write": None,
            "goal": "Request sick leave.",
            "message": "Request sick leave.",
            "intent": "leave_request",
        }
        resolved = app.resolve_pending_auto_goal(
            app.ChatRequest(message="1 day thats all", thread_id=self.thread_id, mode="Auto", history=[]),
            "1 day thats all",
        )
        self.assertIn("Request sick leave.", resolved)
        self.assertIn("1 day thats all", resolved)

    def test_day_first_date_reply_fills_the_date_slot(self):
        app._THREAD_PENDING_GOAL[self.thread_id] = {
            "conversation_id": self.thread_id,
            "status": "waiting_for_fields",
            "missing": "date",
            "missing_fields": ["date"],
            "field_values": {},
            "observations": [],
            "pending_write": None,
            "goal": "Request sick leave.",
            "message": "Request sick leave.",
            "intent": "leave_request",
        }
        resolved = app.resolve_pending_auto_goal(
            app.ChatRequest(message="14th to 15th of august", thread_id=self.thread_id, mode="Auto", history=[]),
            "14th to 15th of august",
        )
        self.assertIn("14th to 15th of august", resolved)
        self.assertNotIn(self.thread_id, app._THREAD_PENDING_GOAL)

    def test_approval_without_a_pending_write_never_calls_gmail(self):
        with patch.object(app.oauth, "post_json") as post_json, patch.object(
            app.general_chat, "respond", return_value="Your email has been sent."
        ) as respond:
            response = self.chat("yes")
        post_json.assert_not_called()
        respond.assert_not_called()
        self.assertIn("nothing to approve", response.reply.lower())

    def test_leave_email_body_mentions_a_date_range_and_volunteered_reason(self):
        plan = orchestrator.AutoPlan(goal="leave", intent="leave_request", steps=[])
        plan.observations = [
            orchestrator.Observation(
                "date_time",
                "resolve_date",
                True,
                "[Date/Time] Resolved date information:\n- 14th to 15th of august: 2026-08-14 (Friday)\n"
                "- 14th to 15th of august: 2026-08-15 (Saturday)",
            )
        ]
        instruction = app.build_leave_email_instruction(
            "request leave 14th to 15th of august to manager@example.com because i am sick",
            plan,
        )
        self.assertIn("August 14, 2026 to August 15, 2026", instruction)
        self.assertIn("Reason: I am sick", instruction)


if __name__ == "__main__":
    unittest.main()
