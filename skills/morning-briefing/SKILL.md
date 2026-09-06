---
name: morning-briefing
description: Executes the morning briefing routine: checks weather for the requested city (default: Moscow) via wttr.in, checks current system date and time, and presents a structured daily overview.
---

# Morning Briefing Routine

This skill provides an automated morning briefing for the user by gathering live data and synthesizing a productive daily digest.

## Steps to Execute

1. **Step 1: Check Weather for Target City**
   - Identify the target city from the user request.
   - If no city is specified, default to `Moscow` (Москва).
   - Call tool `exec` with command:
     ```bash
     curl -s "https://wttr.in/<City>?0&Q&T&lang=ru"
     ```
     *(Example for Moscow: `curl -s "https://wttr.in/Moscow?0&Q&T&lang=ru"`; for London: `curl -s "https://wttr.in/London?0&Q&T&lang=ru"`; for Minsk: `curl -s "https://wttr.in/Minsk?0&Q&T&lang=ru"`).*
   - Capture the minimalist ASCII terminal weather card.

2. **Step 2: Check Current Date & Time**
   - Call tool `exec` with command:
     ```bash
     python -c "import datetime; print(datetime.datetime.now().strftime('%A, %d %B %Y %H:%M'))"
     ```
   - Note current date, day of week, and time.

3. **Step 3: Synthesize Morning Digest**
   - Formulate a cheerful, professional morning report with:
     - 📅 Current date and day of week
     - 🌤 Minimalist ASCII weather widget embedded in a monospace code block (```):
       ```
       <exact ASCII weather output from Step 1>
       ```
     - 💡 Daily productivity recommendation or inspiration
   - Output as `Final Answer: <formatted digest>`
