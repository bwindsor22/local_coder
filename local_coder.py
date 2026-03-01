#!/usr/bin/env python3
"""
local_coder.py — Hardened autonomous coding agent using Ollama.

Safeguards:
  - Cannot finish without a verified file change (no-op write detection)
  - Cannot finish without a successful build (for React/npm projects)
  - JSON retry with error context if the model returns malformed JSON
  - Mandatory read-after-write verification
  - Context compaction when prompt grows too large
  - Build output filtered to error/warning lines only
"""
import os
import re
import sys
import json
import subprocess
import readline  # noqa: F401 — enables arrow-key history in input()
import traceback
from pathlib import Path
from typing import List, Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-coder:30b"

MAX_CONTEXT_CHARS = 120_000
MAX_STEPS = 50
JSON_RETRY_LIMIT = 4   # max consecutive invalid-JSON responses before aborting

# Path to the shared Python tools
TOOLS_DIR = "/Users/brad/projects/code/game-creation-agent/tools"

# Long-term memory file (persists across sessions)
MEMORY_FILE = os.path.expanduser("~/.local_coder_memory.md")


# ============================================================
# Utility
# ============================================================

def print_header(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80 + "\n")


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def list_files_recursive(path: str) -> List[str]:
    results = []
    for root, dirs, files in os.walk(path):
        # Skip common noise directories
        dirs[:] = [d for d in dirs if d not in {
            'node_modules', '.git', '__pycache__', 'build', 'dist', '.next', 'venv', 'env',
            'claude_local_env', 'local_env',
        }]
        for file in files:
            results.append(os.path.join(root, file))
    return results


def filter_build_output(output: str) -> str:
    """Return only error/warning lines from build output."""
    lines = output.splitlines()
    important = []
    for line in lines:
        low = line.lower()
        if any(k in low for k in ('error', 'warn', 'failed', 'cannot', 'unexpected')):
            important.append(line)
    if not important:
        return output[-2000:]  # fallback: last 2000 chars
    return "\n".join(important[-60:])  # last 60 relevant lines


# ============================================================
# Tool Layer
# ============================================================

def tool_read_file(args: dict) -> str:
    path = args["path"]
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", 0))
    content = read_file(path)
    lines = content.splitlines()
    if offset or limit:
        end = offset + limit if limit else None
        lines = lines[offset:end]
        content = "\n".join(lines)
    return content


def tool_write_file(args: dict) -> str:
    path = args["path"]
    # Warn if an absolute path points outside the current working directory
    if os.path.isabs(path):
        cwd = os.getcwd()
        real = os.path.realpath(path)
        if not real.startswith(os.path.realpath(cwd)):
            return (
                f"ERROR: Absolute path '{path}' points outside the project directory '{cwd}'. "
                f"Use a relative path (e.g. 'src/App.js') instead."
            )
    write_file(path, args["content"])
    return f"Wrote {path}"


def tool_list_files(args: dict) -> str:
    return "\n".join(list_files_recursive(args["path"]))


def tool_run_shell(args: dict) -> str:
    command = args["command"]
    cwd = args.get("cwd")
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    out = f"Return code: {result.returncode}\n"
    if result.stdout:
        out += f"STDOUT:\n{result.stdout}\n"
    if result.stderr:
        out += f"STDERR:\n{result.stderr}\n"
    return out


def tool_search_files(args: dict) -> str:
    """Search file contents using ripgrep (falls back to grep)."""
    pattern = args["pattern"]
    path = args.get("path", ".")
    include = args.get("include", "")
    cmd = ["grep", "-r", "-n", "--include", include or "*", pattern, path] if include else \
          ["grep", "-r", "-n", pattern, path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout[:4000] or "(no matches)"


def tool_npm_build(args: dict) -> str:
    project_dir = args["path"]
    sys.path.insert(0, TOOLS_DIR)
    try:
        from npm_build import npm_build  # type: ignore
        result = npm_build(project_dir)
    except ImportError:
        # Fallback: run npm directly
        env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}
        proc = subprocess.run(
            ["/opt/homebrew/bin/npm", "run", "build"],
            cwd=project_dir, capture_output=True, text=True, env=env,
        )
        result = {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as e:
        return f"npm_build error: {e}"

    if result["success"]:
        return "BUILD SUCCESS"
    else:
        combined = result.get("stdout", "") + result.get("stderr", "")
        return "BUILD FAILED\n" + filter_build_output(combined)


def tool_remember(args: dict) -> str:
    """Append a fact to the long-term memory file (~/.local_coder_memory.md)."""
    text = args.get("text", "").strip()
    if not text:
        return "ERROR: 'text' argument required"
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"- {text}\n")
    return f"Saved to memory: {text}"


def tool_git(args: dict) -> str:
    """Run safe read-only or staging git commands."""
    subcmd = args.get("subcommand", "status")
    cwd = args.get("cwd", ".")
    allowed = {"status", "diff", "log", "add", "commit"}
    if subcmd not in allowed:
        return f"git subcommand '{subcmd}' not allowed. Use: {allowed}"
    extra = args.get("args", "")
    full_cmd = f"git {subcmd} {extra}".strip()
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    out = result.stdout + result.stderr
    return out[:3000] or "(empty output)"


TOOLS = {
    "read_file":    tool_read_file,
    "write_file":   tool_write_file,
    "list_files":   tool_list_files,
    "run_shell":    tool_run_shell,
    "search_files": tool_search_files,
    "npm_build":    tool_npm_build,
    "git":          tool_git,
    "remember":     tool_remember,
}

TOOL_SCHEMA = """You must respond ONLY in valid JSON. No markdown, no prose outside the JSON.

Schema:
{
  "thought": "your reasoning about what to do next",
  "action": "tool_name or finish",
  "arguments": { ... },
  "final_answer": "only present when action=finish"
}

Available tools:
- read_file(path, offset?, limit?)          — read a file (optional line range)
- write_file(path, content)                 — create or overwrite a file
- list_files(path)                          — list files recursively (skips node_modules etc.)
- run_shell(command, cwd?)                  — run a shell command
- search_files(pattern, path?, include?)    — grep for a pattern in files
- npm_build(path)                           — build a React/npm project; returns BUILD SUCCESS or BUILD FAILED + errors
- git(subcommand, cwd?, args?)              — run git status/diff/log/add/commit
- remember(text)                            — save a fact to long-term memory for future sessions

Rules:
1. Never invent tool names.
2. FIRST ACTION must always be list_files('.') to see what already exists in the project.
3. Always read a file before writing it (understand existing content).
4. After every write_file you will receive a verification read automatically.
5. Use npm_build after modifying React/JS/CSS files — do not use run_shell for npm.
6. You CANNOT finish until at least one file has been modified (verified non-identical write).
7. For React/npm projects: you ALSO cannot finish until npm_build returns BUILD SUCCESS after your last JS/JSX/CSS change.
8. For Python/non-npm tasks: do NOT call npm_build. Just write files, run_shell to test, then finish.
9. If the build fails, read the error lines carefully, fix the offending file(s), and build again.
10. If you get a JSON parse error, your previous response was invalid — correct it and respond with valid JSON only.
11. Use remember() to save important discoveries: file locations, patterns, gotchas, project conventions.
12. ALWAYS use relative paths (e.g. 'src/App.js'). NEVER use absolute paths — write_file will reject them.
13. If a function or import doesn't exist yet, write that file first before writing code that imports it.
14. Tackle one concrete deliverable at a time. Get npm_build to pass before moving to the next feature.
"""


# ============================================================
# LLM Call
# ============================================================

def call_llm(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 2048},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.Timeout:
        return '{"thought": "LLM timed out", "action": "finish", "final_answer": "ERROR: LLM timed out"}'
    except Exception as e:
        return f'{{"thought": "LLM error: {e}", "action": "finish", "final_answer": "ERROR: {e}"}}'


# ============================================================
# JSON Extraction
# ============================================================

def extract_json(text: str) -> str:
    text = text.strip()
    # Strip leading ``` fences
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    # Find outermost { }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")
    return text[start:end + 1]


def safe_parse(response: str) -> Optional[dict]:
    try:
        cleaned = extract_json(response)
        return json.loads(cleaned)
    except Exception:
        return None


# ============================================================
# Context
# ============================================================

class Context:
    def __init__(self):
        self.messages: List[str] = []
        self.max_tool_chars = 6000

    def _truncate(self, content: str) -> str:
        if len(content) > self.max_tool_chars:
            half = self.max_tool_chars // 2
            return content[:half] + f"\n...[{len(content) - self.max_tool_chars} chars truncated]...\n" + content[-half:]
        return content

    def add(self, role: str, content: str):
        if role.lower() == "tool":
            content = self._truncate(content)
        self.messages.append(f"{role.upper()}:\n{content}\n")

    def build_prompt(self) -> str:
        return "\n".join(self.messages)

    def compact(self):
        print_header("Compacting context...")
        joined = self.build_prompt()

        summary_prompt = f"""Compress this autonomous coding agent conversation into a concise summary.

Return ONLY valid JSON — no prose, no markdown:
{{
  "summary": "2-3 sentence summary of what has been done",
  "modified_files": ["list of file paths that were modified"],
  "current_goal": "exactly what still needs to be done to finish the task",
  "build_status": "passed | failed | not_attempted"
}}

Conversation:
{joined}
"""
        response = call_llm(summary_prompt)
        parsed = safe_parse(response)

        if not parsed:
            print("Compaction failed (invalid JSON). Trimming oldest messages instead.")
            # Keep system messages + last 20 exchanges
            system_msgs = [m for m in self.messages if m.startswith("SYSTEM:")]
            other_msgs = [m for m in self.messages if not m.startswith("SYSTEM:")]
            self.messages = system_msgs + other_msgs[-20:]
            return

        self.messages = []
        summary_text = f"""=== CONTEXT SUMMARY ===
What has been done: {parsed.get('summary', '')}
Modified files: {parsed.get('modified_files', [])}
Build status: {parsed.get('build_status', 'unknown')}
Current goal: {parsed.get('current_goal', '')}
=== END SUMMARY ===
"""
        # Reload persistent context so memory/CLAUDE.md survive compaction
        load_system_context(self, getattr(self, "_project_dir", None))
        self.add("system", summary_text)
        self.add("system", TOOL_SCHEMA)


# ============================================================
# Agent Loop
# ============================================================

def load_system_context(ctx: "Context", project_dir: Optional[str]):
    """Load persistent context: long-term memory and project CLAUDE.md."""
    if os.path.exists(MEMORY_FILE):
        memory = read_file(MEMORY_FILE).strip()
        if memory:
            ctx.add("system", f"Long-term memory (facts from previous sessions):\n{memory}")

    if project_dir:
        claude_md = os.path.join(project_dir, "CLAUDE.md")
        if os.path.exists(claude_md):
            ctx.add("system", f"Project instructions (CLAUDE.md):\n{read_file(claude_md)}")


def run_agent(user_input: str, project_dir: Optional[str] = None):
    ctx = Context()
    ctx._project_dir = project_dir  # store for compact() to reload

    # Change to project directory so relative file paths work correctly
    if project_dir:
        os.chdir(project_dir)

    load_system_context(ctx, project_dir)
    ctx.add("system", TOOL_SCHEMA)
    ctx.add("user", user_input)

    last_write_changed = False  # True once a write that changed content has happened
    build_passed = False        # True once npm_build returned BUILD SUCCESS
    json_fail_count = 0         # Consecutive JSON parse failures
    needs_build = False         # True after any JS/JSX/CSS write in an npm project

    # Only enforce build for npm/React projects
    is_npm_project = bool(
        project_dir and os.path.exists(os.path.join(project_dir, "package.json"))
    )

    for step in range(MAX_STEPS):
        prompt = ctx.build_prompt()
        if len(prompt) > MAX_CONTEXT_CHARS:
            ctx.compact()
            prompt = ctx.build_prompt()

        print(f"\n[Step {step + 1}/{MAX_STEPS}]")
        response = call_llm(prompt)
        print("\nMODEL:", response[:800])

        parsed = safe_parse(response)

        # ── JSON guard ──────────────────────────────────────────────────────
        if not parsed:
            json_fail_count += 1
            if json_fail_count >= JSON_RETRY_LIMIT:
                print(f"\nAborted: {JSON_RETRY_LIMIT} consecutive JSON parse failures.")
                return
            if json_fail_count >= 2:
                # After 2 failures, wipe context and ask for a fresh, simpler approach
                ctx.compact()
                error_msg = (
                    f"REPEATED JSON FAILURE. Your last {json_fail_count} responses were not valid JSON. "
                    f"Stop trying the same approach. Take a step back: list_files('.') to see what exists, "
                    f"then attempt ONE small, concrete action. Respond with ONLY a single valid JSON object."
                )
            else:
                error_msg = (
                    f"Your response was not valid JSON (failure {json_fail_count}/{JSON_RETRY_LIMIT}).\n"
                    f"Failed response (first 200 chars): {response[:200]}\n"
                    f"Respond with ONLY a valid JSON object matching the schema. No markdown, no prose."
                )
            ctx.add("system", error_msg)
            continue
        json_fail_count = 0  # reset on success

        thought = parsed.get("thought", "")
        action = parsed.get("action", "")
        arguments = parsed.get("arguments", {})

        if thought:
            print("\nTHOUGHT:", thought)

        # ── Finish guard ────────────────────────────────────────────────────
        if action == "finish":
            if not last_write_changed:
                ctx.add("system", "Cannot finish: no verified file change has been made yet. Make a change first.")
                continue
            if is_npm_project and needs_build and not build_passed:
                ctx.add("system", "Cannot finish: JS/JSX/CSS files were modified but npm_build has not returned BUILD SUCCESS. Run npm_build.")
                continue
            print_header("DONE")
            print(parsed.get("final_answer", "(no summary)"))
            return

        if action not in TOOLS:
            ctx.add("system", f"Unknown action '{action}'. Valid actions: {list(TOOLS.keys())} or 'finish'.")
            continue

        # ── Write file (with before/after verification) ──────────────────
        if action == "write_file":
            path = arguments.get("path", "")
            try:
                before = read_file(path)
            except FileNotFoundError:
                before = None

            try:
                result = TOOLS[action](arguments)
            except Exception as e:
                ctx.add("system", f"write_file failed: {e}")
                continue

            try:
                after = read_file(path)
            except Exception as e:
                ctx.add("system", f"Verification read failed: {e}")
                continue

            if before == after:
                ctx.add("system", "File content did not change (no-op write). The write had no effect — review your content.")
                last_write_changed = False
                continue

            last_write_changed = True
            # Only require a rebuild for JS/JSX/CSS/TS files in an npm project
            if is_npm_project and any(path.endswith(ext) for ext in ('.js', '.jsx', '.ts', '.tsx', '.css')):
                needs_build = True
                build_passed = False  # must rebuild after any JS/CSS change

            ctx.add("assistant", response)
            ctx.add("tool", result)
            ctx.add("tool", f"Verification — file after write (first 60 lines):\n" + "\n".join(after.splitlines()[:60]))
            print(f"\n✓ wrote {path}")
            continue

        # ── npm_build ─────────────────────────────────────────────────────
        if action == "npm_build":
            try:
                result = TOOLS[action](arguments)
            except Exception as e:
                result = f"npm_build exception: {e}\n{traceback.format_exc()}"

            ctx.add("assistant", response)
            ctx.add("tool", result)
            print("\nBUILD RESULT:", result[:300])

            if result.startswith("BUILD SUCCESS"):
                build_passed = True
                needs_build = False
                ctx.add("system", "Build passed. You may now finish if the task is complete.")
            else:
                build_passed = False
                ctx.add("system", "Build failed. Read the errors above, fix the offending file(s), then run npm_build again.")
            continue

        # ── All other tools ───────────────────────────────────────────────
        try:
            result = TOOLS[action](arguments)
        except Exception as e:
            result = f"Tool error: {e}\n{traceback.format_exc()}"

        ctx.add("assistant", response)
        ctx.add("tool", result)
        print(f"\nTOOL [{action}]:", str(result)[:200])

    print(f"\nMax steps ({MAX_STEPS}) reached. Task incomplete.")


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hardened local coding agent (Ollama)")
    parser.add_argument("--project", "-p", default=None, help="Project directory (loads CLAUDE.md if present)")
    parser.add_argument("--task", "-t", default=None, help="Run a single task non-interactively")
    args = parser.parse_args()

    if args.task:
        run_agent(args.task, project_dir=args.project)
        return

    print_header("Local Coder — Hardened Agent")
    print(f"Model: {MODEL}")
    print(f"Project: {args.project or '(none)'}")
    print("Type /exit to quit, /help for commands\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/help":
            print("Commands: /exit  /help  /memory  /forget")
            print("Or enter a coding task and the agent will work on it.")
            continue
        if user_input == "/memory":
            if os.path.exists(MEMORY_FILE):
                print(read_file(MEMORY_FILE) or "(memory file is empty)")
            else:
                print("(no memory file yet)")
            continue
        if user_input == "/forget":
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
                print("Memory cleared.")
            else:
                print("(no memory file to clear)")
            continue

        run_agent(user_input, project_dir=args.project)


if __name__ == "__main__":
    main()
