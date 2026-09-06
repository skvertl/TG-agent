---
name: system-health
description: Executes the system and container diagnostics routine: inspects host/container uptime, disk and RAM usage, verifies Ollama API connectivity, and produces an operational health status card.
---

# System Health & Container Diagnostics Routine

This skill inspects runtime infrastructure health, system resource limits, and AI inference backend availability.

## Steps to Execute

1. **Step 1: Check System Memory & Platform**
   - Call tool `exec` with command:
     ```bash
     python -c "import platform, sys; print(f'OS: {platform.system()} {platform.release()}, Python: {sys.version.split()[0]}')"
     ```
   - On Linux/Docker, also inspect memory: `free -m` or `python -c "import os; print(os.uname())"`.

2. **Step 2: Check Disk Space**
   - Call tool `exec` with command:
     ```bash
     python -c "import shutil; total, used, free = shutil.disk_usage('.'); print(f'Disk: Used {used // (2**30)}GB / Total {total // (2**30)}GB ({free // (2**30)}GB free)')"
     ```

3. **Step 3: Check Ollama API Connectivity**
   - Call tool `exec` with command:
     ```bash
     curl -s -m 5 http://localhost:11434/api/version
     ```
     *(Or if in Docker: `curl -s -m 5 http://host.docker.internal:11434/api/version`)*.

4. **Step 4: Synthesize System Health Card**
   - Present a concise, structured operational card:
     - 🖥 **System & OS:** Platform details.
     - 💾 **Disk & RAM:** Disk usage and resource status.
     - 🤖 **Ollama Engine:** Version / Connectivity (Online or Offline).
     - 🟢 **Overall Verdict:** Healthy / Warning / Degraded.
   - Output as `Final Answer: <formatted card>`.
