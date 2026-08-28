"""
google_calendar_tool.py
========================
Google Calendar MCP-style tool. Schedule checks are read-only. Creating an
event is still a write action and always asks for approval first.
"""

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

from . import oauth
from .base import MCPTool, ToolAction

CALENDAR_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"


@dataclass
class CalendarEventRequest:
    summary: str
    start: datetime
    end: datetime
    attendee_email: str


class GoogleCalendarTool(MCPTool):
    def call(self, action_name: str, confirmed: bool = False, **kwargs) -> str:
        action = next((a for a in self.actions if a.name == action_name), None)
        if action is None:
            return f"[{self.display_name}] Unknown action '{action_name}'."

        if not self.connected:
            return "Google Calendar isn't connected. Connect it in MCP Connections and I can check your schedule."

        if action_name == "check_schedule":
            return self._check_schedule(kwargs.get("instruction", ""))

        if action_name == "create_meeting":
            return self._create_meeting(
                confirmed=confirmed,
                instruction=kwargs.get("instruction", ""),
                event_request=kwargs.get("event_request"),
            )

        return super().call(action_name, confirmed=confirmed, **kwargs)

    def _check_schedule(self, instruction: str) -> str:
        start, end, label = _calendar_window(instruction)
        access_token = oauth.google_access_token()
        params = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": "10",
        }
        payload = oauth.get_json(f"{CALENDAR_EVENTS_URL}?{urlencode(params)}", access_token)
        events = payload.get("items", [])
        if not events:
            return f"[Google Calendar] No events found for {label}."

        lines = [f"[Google Calendar] Events for {label}:"]
        for event in events:
            summary = event.get("summary") or "(Untitled event)"
            start_info = event.get("start", {})
            end_info = event.get("end", {})
            start_text = _format_event_time(start_info.get("dateTime") or start_info.get("date"))
            end_text = _format_event_time(end_info.get("dateTime") or end_info.get("date"))
            when = start_text if not end_text else f"{start_text} - {end_text}"
            lines.append(f"- {when}: {summary}")
        return "\n".join(lines)

    def _create_meeting(self, confirmed: bool, instruction: str, event_request=None) -> str:
        event_request = coerce_calendar_event_request(event_request) or parse_calendar_event_request(instruction)
        if not event_request:
            return "[Google Calendar] I need a date, time, and attendee email before I can prepare that meeting."

        preview = (
            "[Google Calendar] This will create a real calendar event:\n\n"
            f"Title: {event_request.summary}\n"
            f"When: {event_request.start.isoformat()} - {event_request.end.isoformat()}\n"
            f"Attendee: {event_request.attendee_email}\n\n"
            "Please confirm before it runs."
        )
        if not confirmed:
            return preview

        access_token = oauth.google_access_token()
        payload = build_calendar_event_payload(event_request)
        response = oauth.post_json(CALENDAR_EVENTS_URL, access_token, payload)
        event_id = response.get("id", "unknown")
        link = response.get("htmlLink")
        suffix = f" Link: {link}" if link else ""
        return f"[Google Calendar] Created meeting '{event_request.summary}'. Calendar event id: {event_id}.{suffix}"


def coerce_calendar_event_request(value) -> Optional[CalendarEventRequest]:
    if isinstance(value, CalendarEventRequest):
        return value
    if isinstance(value, dict):
        try:
            start = value.get("start")
            end = value.get("end")
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            if isinstance(end, str):
                end = datetime.fromisoformat(end)
            summary = str(value.get("summary") or "").strip()
            attendee_email = str(value.get("attendee_email") or "").strip()
            if summary and isinstance(start, datetime) and isinstance(end, datetime) and attendee_email:
                return CalendarEventRequest(summary=summary, start=start, end=end, attendee_email=attendee_email)
        except ValueError:
            return None
    return None


def parse_calendar_event_request(instruction: str) -> Optional[CalendarEventRequest]:
    text = " ".join((instruction or "").strip().split())
    if not text:
        return None

    attendee = ""
    attendee_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    if attendee_match:
        attendee = attendee_match.group(0)
    if not attendee:
        return None

    now = datetime.now().astimezone()
    day = _parse_event_day(text, now)
    hour_minute = _parse_event_time(text)
    if not day or not hour_minute:
        return None

    hour, minute = hour_minute
    start = datetime.combine(day, time(hour=hour, minute=minute), tzinfo=now.tzinfo)
    end = start + timedelta(minutes=30)
    summary = _parse_event_summary(text, attendee)
    return CalendarEventRequest(summary=summary, start=start, end=end, attendee_email=attendee)


def build_calendar_event_payload(event_request: CalendarEventRequest) -> dict:
    return {
        "summary": event_request.summary,
        "start": {"dateTime": event_request.start.isoformat()},
        "end": {"dateTime": event_request.end.isoformat()},
        "attendees": [{"email": event_request.attendee_email}],
    }


def _calendar_window(instruction: str) -> tuple[datetime, datetime, str]:
    now = datetime.now(timezone.utc)
    text = (instruction or "").lower()
    if "tomorrow" in text:
        day = (now + timedelta(days=1)).date()
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        return start, start + timedelta(days=1), "tomorrow"
    if "today" in text:
        day = now.date()
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        return start, start + timedelta(days=1), "today"
    return now, now + timedelta(days=7), "the next 7 days"


def _parse_event_day(text: str, now: datetime):
    lowered = text.lower()
    if "today" in lowered:
        return now.date()
    if "tomorrow" in lowered:
        return (now + timedelta(days=1)).date()
    weekday_lookup = {
        "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
    }
    weekday_match = re.search(r"\b(?:(next|this)\s+)?(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:r|rs|rsday|rday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b", lowered)
    if weekday_match:
        weekday = weekday_lookup.get(weekday_match.group(2))
        if weekday is not None:
            days = (weekday - now.weekday()) % 7
            if weekday_match.group(1) == "next" and days == 0:
                days = 7
            return (now + timedelta(days=days)).date()
    date_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lowered)
    if date_match:
        return datetime(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))).date()
    slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", lowered)
    if slash_match:
        year = slash_match.group(3)
        if not year:
            year = str(now.year)
        elif len(year) == 2:
            year = f"20{year}"
        return datetime(int(year), int(slash_match.group(1)), int(slash_match.group(2))).date()
    return None


def _parse_event_time(text: str) -> Optional[tuple[int, int]]:
    lowered = text.lower()
    if "noon" in lowered:
        return 12, 0
    if "midnight" in lowered:
        return 0, 0
    match = re.search(r"\b(?:at|from)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lowered)
    if not match:
        match = re.search(r"\b(?:at|from)\s+(\d{1,2})(?::(\d{2}))?\b", lowered)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3) if len(match.groups()) >= 3 else None
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _parse_event_summary(text: str, attendee_email: str) -> str:
    subject_match = re.search(r"\b(?:title|subject)\s+['\"]?(.+?)['\"]?(?:\s+(?:with|at|on|tomorrow|today|next|this)\b|$)", text, re.IGNORECASE)
    if subject_match:
        return subject_match.group(1).strip(" .'\"")
    return f"Meeting with {attendee_email}"


def _format_event_time(value: str) -> str:
    if not value:
        return ""
    if "T" not in value:
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


google_calendar_tool = GoogleCalendarTool(
    key="google_calendar",
    display_name="Google Calendar",
    icon="📅",
    description="Check your schedule and create meetings (with approval).",
    actions=[
        ToolAction("check_schedule", "Check upcoming events", write_action=False),
        ToolAction("create_meeting", "Create a calendar event", write_action=True),
    ],
)
