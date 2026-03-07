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
MAX_STEPS = 80
JSON_RETRY_LIMIT = 4   # max consecutive invalid-JSON responses before aborting

# Path to the shared Python tools
TOOLS_DIR = "/Users/brad/projects/code/game-creation-agent/tools"

# Long-term memory: global fallback; project memory preferred when project_dir is set
GLOBAL_MEMORY_FILE = os.path.expanduser("~/.local_coder_memory.md")

def get_memory_file(project_dir: Optional[str] = None) -> str:
    """Return the active memory file: project-scoped if available, else global."""
    if project_dir:
        return os.path.join(project_dir, ".local_coder_memory.md")
    return GLOBAL_MEMORY_FILE


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
    return "\n".join(list_files_recursive(args.get("path", ".")))


def tool_run_shell(args: dict) -> str:
    command = args["command"]
    cwd = args.get("cwd")
    timeout_seconds = int(args.get("timeout_seconds", 60))
    timeout_seconds = min(timeout_seconds, 600)  # cap at 10 minutes
    # Extend PATH to include common binary locations (homebrew, local)
    env = {
        **os.environ,
        "PATH": f"/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}",
    }
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: command exceeded {timeout_seconds}s. For long-running processes use 'command &' to run in background."
    out = f"Return code: {result.returncode}\n"
    if result.stdout:
        out += f"STDOUT:\n{result.stdout[:3000]}\n"
    if result.stderr:
        out += f"STDERR:\n{result.stderr[:1000]}\n"
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
    """Append a fact to the project memory file (or global if no project)."""
    text = args.get("text", "").strip()
    if not text:
        return "ERROR: 'text' argument required"
    mem_file = get_memory_file(args.get("_project_dir"))
    with open(mem_file, "a", encoding="utf-8") as f:
        f.write(f"- {text}\n")
    return f"Saved to memory ({mem_file}): {text}"


def tool_screenshot(args: dict) -> str:
    """Take a screenshot of a URL and return a visual description via vision_tool."""
    url = args.get("url", "http://localhost:3000")
    script = (
        f"import sys; sys.path.insert(0, {repr(TOOLS_DIR)}); "
        f"from screenshot_tool import take_screenshot; "
        f"from vision_tool import ask_about_screenshot; "
        f"path = take_screenshot({repr(url)}, output_path='/tmp/lc_screenshot.png'); "
        f"print(ask_about_screenshot(path, 'Describe what you see on the page. Note any visible UI elements, text, errors, or layout issues.'))"
    )
    result = subprocess.run(
        ["/usr/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return f"screenshot error: {result.stderr[:400]}"
    return result.stdout.strip()


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


def tool_edit_file(args: dict) -> str:
    """Replace old_str with new_str in a file (targeted patch, not full rewrite)."""
    path = args["path"]
    old_str = args["old_str"]
    new_str = args["new_str"]
    replace_all = args.get("replace_all", False)

    if os.path.isabs(path):
        cwd = os.getcwd()
        real = os.path.realpath(path)
        if not real.startswith(os.path.realpath(cwd)):
            return (
                f"ERROR: Absolute path '{path}' points outside the project directory '{cwd}'. "
                f"Use a relative path instead."
            )
    try:
        content = read_file(path)
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"

    if old_str not in content:
        # Find nearby lines to help model self-correct
        first_line = old_str.strip().split('\n')[0][:30]
        simple = re.sub(r'[^a-zA-Z0-9 ]', '', first_line).lower()
        hint_lines = []
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if simple and re.sub(r'[^a-zA-Z0-9 ]', '', line).lower().find(simple) >= 0:
                hint_lines = lines[max(0, i-1):i+3]
                break
        hint = ('\nNearest matching lines:\n' + '\n'.join(hint_lines)) if hint_lines else ''
        return (
            f"ERROR: old_str not found verbatim in {path}. "
            f"Unicode characters like em-dash (—) differ from hyphen (-) — they must match exactly. "
            f"Call read_file('{path}') to get exact text, then copy it precisely.{hint}"
        )

    count = content.count(old_str)
    if not replace_all and count > 1:
        return (
            f"ERROR: old_str appears {count} times in {path}. "
            f"Add more surrounding context to make it unique, or set replace_all=true."
        )

    new_content = content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1)
    write_file(path, new_content)
    replaced = "all occurrences" if replace_all else "1 occurrence"
    return f"Edited {path}: replaced {replaced}."


def tool_glob(args: dict) -> str:
    """Find files matching a glob pattern (e.g. 'src/**/*.jsx')."""
    import glob as glob_module
    pattern = args["pattern"]
    base = args.get("base", ".")
    full_pattern = os.path.join(base, pattern) if not os.path.isabs(pattern) else pattern
    matches = sorted(glob_module.glob(full_pattern, recursive=True))
    noise = {'node_modules', '.git', '__pycache__', 'build', 'dist', '.next', 'venv', 'env'}
    filtered = [m for m in matches if not any(n in m.split(os.sep) for n in noise)]
    return "\n".join(filtered) or "(no matches)"


def tool_read_pdf(args: dict) -> str:
    """Extract text from a PDF rulebook. pages: '1-5' or '3' (optional)."""
    path = args.get("path")
    pages = args.get("pages")
    if not path:
        return "ERROR: path required"
    script = (
        f"import sys; sys.path.insert(0, {repr(TOOLS_DIR)}); "
        f"from pdf_tool import pdf_to_text; "
        f"print(pdf_to_text({repr(path)}, pages={repr(pages)})[:8000])"
    )
    result = subprocess.run(
        ["/usr/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return f"read_pdf error: {result.stderr[:400]}"
    return result.stdout.strip()


TOOLS = {
    "read_file":    tool_read_file,
    "write_file":   tool_write_file,
    "list_files":   tool_list_files,
    "run_shell":    tool_run_shell,
    "search_files": tool_search_files,
    "edit_file":    tool_edit_file,
    "glob":         tool_glob,
    "npm_build":    tool_npm_build,
    "git":          tool_git,
    "read_pdf":     tool_read_pdf,
    "remember":     tool_remember,
    "screenshot":   tool_screenshot,
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
- edit_file(path, old_str, new_str, replace_all?) — patch a file: replace old_str with new_str
- write_file(path, content)                 — create or fully overwrite a file (use only for new files or complete rewrites)
- glob(pattern, base?)                      — find files by glob pattern (e.g. 'src/**/*.jsx')
- list_files(path)                          — list all files recursively (skips node_modules etc.)
- run_shell(command, cwd?, timeout_seconds?) — run a shell command (default 60s timeout; use 'cmd &' for background)
- search_files(pattern, path?, include?)    — grep for a pattern in files
- npm_build(path)                           — build a React/npm project; returns BUILD SUCCESS or BUILD FAILED + errors. ALWAYS pass path="." to build the current project.
- git(subcommand, cwd?, args?)              — run git status/diff/log/add/commit
- read_pdf(path, pages?)                    — extract text from a PDF rulebook
- remember(text)                            — save a fact to long-term memory for future sessions
- screenshot(url)                           — take a screenshot of a URL and return a visual description

Rules:
1. Never invent tool names.
2. FIRST ACTION must always be list_files('.') to see what already exists in the project. Do NOT skip this — even if you think you know the structure.
3. Always read a file before modifying it (understand existing content).
4. PREFER edit_file over write_file for changes to existing files — it's safer and uses less context.
5. Use write_file only when creating a new file or completely rewriting an existing one.
6. After every write_file you will receive a verification read automatically.
7. Use npm_build(path=".") after modifying React/JS/CSS files — always pass "." to build the current project. NEVER use an absolute path like "/Users/brad/..." for npm_build.
8. You CANNOT finish until at least one file has been modified (verified non-identical write). Verifying a build or reading files does NOT count as completing the task.
9. For React/npm projects: you ALSO cannot finish until npm_build returns BUILD SUCCESS after your last JS/JSX/CSS change.
10. For Python/non-npm tasks: do NOT call npm_build. Just write files, run_shell to test, then finish.
10b. For Node.js scripts that are NOT React components (e.g. trainer scripts, CLI tools): do NOT call npm_build. Use run_shell to test ('node script.js' or 'node script.js &' for background).
10c. For long-running commands (training, dev servers): append '&' to run in background. run_shell will return immediately.
10d. Do not wait for training runs to complete — start them in background, then finish the task.
11. If the build fails, read the error lines carefully, fix the offending file(s), and build again.
12. If you get a JSON parse error, your previous response was invalid — correct it and respond with valid JSON only.
13. Use remember() to save important discoveries: file locations, patterns, gotchas, project conventions.
14. ALWAYS use relative paths (e.g. 'src/App.js'). NEVER use absolute paths — write_file/edit_file will reject them.
17. If the task references specific files that don't exist, immediately call list_files('.') and report which files are missing BEFORE attempting any implementation. Do not silently invent an alternative approach.
15. If a function or import doesn't exist yet, write that file first before writing code that imports it.
16. Tackle one concrete deliverable at a time. Get npm_build to pass before moving to the next feature.
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
                "options": {"temperature": 0.1, "num_predict": 4096},
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
    # Find the first complete JSON object using balanced brace counting.
    # rfind("}") would fail when the model echoes the object twice in one reply.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("Unbalanced braces — no complete JSON object found")


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
        # Always pin the original task so the model can't drift after compaction
        original_task = getattr(self, "_original_task", None)
        if original_task:
            self.add("system", f"PINNED TASK — your ONLY goal, do not deviate:\n{original_task}")
        self.add("system", summary_text)
        self.add("system", TOOL_SCHEMA)


# ============================================================
# Agent Loop
# ============================================================

def load_system_context(ctx: "Context", project_dir: Optional[str]):
    """Load persistent context: project memory, CLAUDE.md, then global memory."""
    if project_dir:
        # Project-scoped memory takes precedence
        proj_mem = os.path.join(project_dir, ".local_coder_memory.md")
        if os.path.exists(proj_mem):
            memory = read_file(proj_mem).strip()
            if memory:
                ctx.add("system", f"Project memory (facts about this project):\n{memory}")
        claude_md = os.path.join(project_dir, "CLAUDE.md")
        if os.path.exists(claude_md):
            ctx.add("system", f"Project instructions (CLAUDE.md):\n{read_file(claude_md)}")
    else:
        # No project: fall back to global memory
        if os.path.exists(GLOBAL_MEMORY_FILE):
            memory = read_file(GLOBAL_MEMORY_FILE).strip()
            if memory:
                ctx.add("system", f"Long-term memory:\n{memory}")


def make_plan(user_input: str, project_dir: Optional[str]) -> str:
    """
    Ask the model to produce a numbered step-by-step plan before execution.
    Returns the plan string (injected as pinned system context).
    """
    ctx = Context()
    load_system_context(ctx, project_dir)
    ctx.add("user", (
        f"You are about to work on this task:\n\n{user_input}\n\n"
        "Before writing any code, output ONLY a numbered plan (5-10 steps) of exactly what files "
        "you will read, what changes you will make, and how you will verify the result. "
        "Be specific about file names and what text to find/replace. "
        "No JSON — plain text list only."
    ))
    response = call_llm(ctx.build_prompt())
    # Strip any JSON wrapper if model slips into it
    plan = response.strip()
    if plan.startswith("{"):
        plan = "(plan generation failed — proceeding without plan)"
    return plan


def self_assess(task: str, modified_files: list, ctx_snapshot: str) -> str:
    """
    Ask the model to score its own work on three dimensions before finishing.
    Returns an assessment dict string: {"score": int, "issues": [...], "verdict": "pass|retry"}.
    Score 0-10. If score < 7 or verdict=="retry", the agent loop should continue.
    """
    prompt = (
        f"You just attempted this task:\n{task}\n\n"
        f"Files modified: {modified_files}\n\n"
        f"Conversation so far (last 3000 chars):\n{ctx_snapshot[-3000:]}\n\n"
        "Rate the outcome on a scale 0-10 across:\n"
        "  1. Correctness — does the code actually do what was asked?\n"
        "  2. Completeness — are all parts of the task done?\n"
        "  3. Build status — did the build pass?\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"score": <0-10>, "issues": ["list any problems"], "verdict": "pass" or "retry"}\n'
        "verdict=retry if score < 7 or there are unresolved issues."
    )
    response = call_llm(prompt)
    parsed = safe_parse(response)
    if not parsed:
        return '{"score": 5, "issues": ["self-assessment parse failed"], "verdict": "pass"}'
    return json.dumps(parsed)


def run_agent(user_input: str, project_dir: Optional[str] = None, planning: bool = True, self_assessment: bool = True):
    ctx = Context()
    ctx._project_dir = project_dir  # store for compact() to reload
    ctx._original_task = user_input  # pinned so it survives compaction

    # ── Planning phase ────────────────────────────────────────────────────────
    if planning:
        print_header("Planning phase")
        plan = make_plan(user_input, project_dir)
        print(plan)
        ctx._plan = plan
    else:
        plan = None

    # Change to project directory so relative file paths work correctly
    if project_dir:
        os.chdir(project_dir)

    load_system_context(ctx, project_dir)
    # Inject plan as pinned context so execution stays on track
    if plan:
        ctx.add("system", f"EXECUTION PLAN — follow these steps in order:\n{plan}")
    ctx.add("system", TOOL_SCHEMA)
    ctx.add("user", user_input)

    last_write_changed = False  # True once a write that changed content has happened
    build_passed = False        # True once npm_build returned BUILD SUCCESS
    json_fail_count = 0         # Consecutive JSON parse failures
    needs_build = False         # True after any JS/JSX/CSS write in an npm project
    modified_files: list = []

    # Only enforce build for npm/React projects
    is_npm_project = bool(
        project_dir and os.path.exists(os.path.join(project_dir, "package.json"))
    )

    for step in range(MAX_STEPS):
        compacted_this_step = False
        prompt = ctx.build_prompt()
        if len(prompt) > MAX_CONTEXT_CHARS:
            ctx.compact()
            compacted_this_step = True
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
                # After 2 failures, wipe context and ask for a fresh, simpler approach.
                # Skip if we already compacted this step to avoid double-compaction.
                if not compacted_this_step:
                    ctx.compact()
                    compacted_this_step = True  # noqa: F841
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
                ctx.add("system",
                    "Cannot finish: no verified file change has been made yet. "
                    "Verifying a build or reading files does NOT count. "
                    "Your edit_file/write_file calls have either not been attempted or returned an ERROR. "
                    "Re-read the target file, copy the EXACT text to replace (including Unicode chars), and call edit_file again.")
                continue
            if is_npm_project and needs_build and not build_passed:
                ctx.add("system", "Cannot finish: JS/JSX/CSS files were modified but npm_build has not returned BUILD SUCCESS. Run npm_build.")
                continue
            # ── Self-assessment before completion ─────────────────────────
            if self_assessment and modified_files:
                print_header("Self-assessment")
                assessment_str = self_assess(user_input, modified_files, ctx.build_prompt())
                print(assessment_str)
                assessment = safe_parse(assessment_str) or {}
                score = assessment.get("score", 10)
                verdict = assessment.get("verdict", "pass")
                issues = assessment.get("issues", [])
                if verdict == "retry" or score < 7:
                    retry_msg = (
                        f"Self-assessment score: {score}/10. Verdict: {verdict}. "
                        f"Issues: {issues}. "
                        "Do NOT finish yet — address the issues above, then try again."
                    )
                    ctx.add("system", retry_msg)
                    continue
                else:
                    print(f"  Score {score}/10 — passing.")
            print_header("DONE")
            print(parsed.get("final_answer", "(no summary)"))
            return

        if action not in TOOLS:
            ctx.add("system", f"Unknown action '{action}'. Valid actions: {list(TOOLS.keys())} or 'finish'.")
            continue

        # ── Write / edit file (with before/after verification) ──────────────────
        if action in ("write_file", "edit_file"):
            path = arguments.get("path", "")
            try:
                before = read_file(path)
            except FileNotFoundError:
                before = None

            try:
                result = TOOLS[action](arguments)
            except Exception as e:
                ctx.add("system", f"{action} failed: {e}")
                continue

            # If edit_file returned an ERROR string, surface it without verifying
            if action == "edit_file" and isinstance(result, str) and result.startswith("ERROR"):
                ctx.add("assistant", response)
                ctx.add("system", result)
                print(f"\n✗ {action} error: {result[:120]}")
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
            if path not in modified_files:
                modified_files.append(path)
            # Only require a rebuild for React/browser JS files (not Node-only trainer scripts)
            # Node.js scripts in src/AI/, scripts/, tools/ don't affect the React bundle
            _is_react_file = is_npm_project and (
                any(path.endswith(ext) for ext in ('.jsx', '.tsx', '.css')) or
                (any(path.endswith(ext) for ext in ('.js', '.ts')) and
                 not any(seg in path for seg in ('AI/', 'ai/', 'scripts/', 'tools/', 'trainer')))
            )
            if _is_react_file:
                needs_build = True
                build_passed = False  # must rebuild after any React JS/CSS change

            ctx.add("assistant", response)
            ctx.add("tool", result)
            ctx.add("tool", f"Verification — file after {action} (first 60 lines):\n" + "\n".join(after.splitlines()[:60]))
            print(f"\n✓ {action} {path}")
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
            # Inject project_dir so remember() knows where to write
            if action == "remember":
                arguments = {**arguments, "_project_dir": project_dir}
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
    parser.add_argument("--no-plan", action="store_true", help="Skip planning phase")
    parser.add_argument("--no-assess", action="store_true", help="Skip self-assessment before finish")
    args = parser.parse_args()

    if args.task:
        run_agent(args.task, project_dir=args.project,
                  planning=not args.no_plan, self_assessment=not args.no_assess)
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
            mem = get_memory_file(args.project)
            if os.path.exists(mem):
                print(read_file(mem) or "(memory file is empty)")
            else:
                print("(no memory file yet)")
            continue
        if user_input == "/forget":
            mem = get_memory_file(args.project)
            if os.path.exists(mem):
                os.remove(mem)
                print("Memory cleared.")
            else:
                print("(no memory file to clear)")
            continue

        run_agent(user_input, project_dir=args.project)


if __name__ == "__main__":
    main()
