"""
Автоматическая синхронизация глобальных скиллов с AGENTS.md.
Считывает YAML-шапки (frontmatter) из всех SKILL.md и обновляет каталог в AGENTS.md.
"""
import os
import re
from pathlib import Path

GLOBAL_SKILLS_DIR = Path.home() / ".gemini" / "config" / "skills"
AGENTS_MD_PATH = Path(__file__).resolve().parent.parent / "AGENTS.md"

CATALOG_START = "<!-- SKILLS_CATALOG_START -->"
CATALOG_END = "<!-- SKILLS_CATALOG_END -->"


def parse_skill_frontmatter(skill_path: Path) -> dict:
    """Извлекает name и description из YAML frontmatter SKILL.md."""
    try:
        content = skill_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}
        
        in_fm = False
        name = ""
        desc_lines = []
        in_desc = False

        for line in lines:
            if line.strip() == "---":
                if not in_fm:
                    in_fm = True
                    continue
                else:
                    break
            if in_fm:
                if line.startswith("name:"):
                    name = line.split("name:", 1)[1].strip()
                    in_desc = False
                elif line.startswith("description:"):
                    desc_lines.append(line.split("description:", 1)[1].strip())
                    in_desc = True
                elif in_desc:
                    if line.startswith(" ") or line.startswith("\t"):
                        desc_lines.append(line.strip())
                    else:
                        in_desc = False

        return {
            "name": name or skill_path.parent.name,
            "description": " ".join(desc_lines).strip(),
            "path": str(skill_path),
        }
    except Exception as e:
        print(f"Ошибка чтения {skill_path}: {e}")
        return {}


def generate_catalog_markdown() -> str:
    """Формирует Markdown-секцию со всеми доступными скиллами."""
    skills = []
    if GLOBAL_SKILLS_DIR.exists():
        for skill_dir in sorted(GLOBAL_SKILLS_DIR.iterdir()):
            skill_file = skill_dir / "SKILL.md"
            if skill_file.is_file():
                meta = parse_skill_frontmatter(skill_file)
                if meta:
                    skills.append(meta)

    lines = [
        CATALOG_START,
        "### Available Skills & Autonomous Activation Catalog",
        "",
        "> [!IMPORTANT]",
        "> Before executing any non-trivial task, the agent **MUST consult this catalog**.",
        "> If the task intent matches the scope of a skill, the agent **MUST activate it** via `view_file` before writing code.",
        "",
        "| Skill | Purpose & Trigger Condition |",
        "| :--- | :--- |",
    ]

    for s in skills:
        name = s["name"]
        desc = s["description"]
        lines.append(f"| **`{name}`** | {desc} |")

    lines.append("")
    lines.append(CATALOG_END)
    return "\n".join(lines)


def sync_agents_md():
    if not AGENTS_MD_PATH.exists():
        print(f"Файл {AGENTS_MD_PATH} не найден.")
        return

    content = AGENTS_MD_PATH.read_text(encoding="utf-8")
    new_catalog = generate_catalog_markdown()

    if CATALOG_START in content and CATALOG_END in content:
        pattern = re.compile(
            rf"{re.escape(CATALOG_START)}.*?{re.escape(CATALOG_END)}",
            re.DOTALL,
        )
        updated_content = pattern.sub(new_catalog, content)
    else:
        # Insert catalog into section 1
        anchor = "## 1. Autonomous Skill Usage"
        if anchor in content:
            updated_content = content.replace(
                anchor,
                f"{anchor}\n\n{new_catalog}",
                1,
            )
        else:
            updated_content = content + "\n\n" + new_catalog

    AGENTS_MD_PATH.write_text(updated_content, encoding="utf-8")
    print(f"Каталог скиллов успешно синхронизирован в {AGENTS_MD_PATH}")


if __name__ == "__main__":
    sync_agents_md()
