#!/usr/bin/env python3
"""
sync_tracker.py - Utility script to sync Git commits and agent updates into track.html
"""

import sys
import os
import re
import json
import subprocess
from datetime import datetime

TRACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track.html")

def extract_json_from_html(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'<script id="agent-tracking-data" type="application/json">(.*?)</script>', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find <script id=\"agent-tracking-data\"> in track.html")
    return json.loads(match.group(1).strip()), content, match.span(1)

def save_json_to_html(html_path, full_content, span, data):
    new_json_str = "\n" + json.dumps(data, indent=2) + "\n"
    updated_content = full_content[:span[0]] + new_json_str + full_content[span[1]:]
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

def get_git_commits(limit=15):
    try:
        # Check if git repo
        res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return []
        
        cmd = ["git", "log", f"-n{limit}", "--pretty=format:%H|||%an <%ae>|||%ad|||%s", "--date=iso-strict"]
        out = subprocess.check_output(cmd, text=True).strip()
        if not out:
            return []
        
        commits = []
        for line in out.splitlines():
            parts = line.split("|||")
            if len(parts) == 4:
                commit_hash, author, date_str, msg = parts
                # Get modified files for this commit
                files_out = subprocess.check_output(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash], text=True).strip()
                files = [f for f in files_out.splitlines() if f]
                commits.append({
                    "hash": commit_hash[:8],
                    "timestamp": date_str,
                    "author": author,
                    "message": msg,
                    "files": files or ["(root)"]
                })
        return commits
    except Exception as e:
        print(f"[Warning] Git error: {e}")
        return []

def sync_git():
    data, full_content, span = extract_json_from_html(TRACK_FILE)
    git_commits = get_git_commits()
    if git_commits:
        data["commits"] = git_commits
        save_json_to_html(TRACK_FILE, full_content, span, data)
        print(f"Successfully synced {len(git_commits)} git commit(s) into track.html.")
    else:
        print("No git commits found to sync.")

def add_prompt(prompt_text, source="Agent / Subagent", notes="", status="Logged"):
    data, full_content, span = extract_json_from_html(TRACK_FILE)
    next_id = f"PROMPT-{len(data.get('idea_prompts', [])) + 1:03d}"
    timestamp = datetime.now().astimezone().isoformat()
    new_prompt = {
        "id": next_id,
        "timestamp": timestamp,
        "source": source,
        "prompt": prompt_text,
        "status": status,
        "notes": notes
    }
    data.setdefault("idea_prompts", []).append(new_prompt)
    save_json_to_html(TRACK_FILE, full_content, span, data)
    print(f"Added new prompt {next_id} to track.html.")

def add_milestone(title, description, status="Pending", date_str=None):
    data, full_content, span = extract_json_from_html(TRACK_FILE)
    next_id = f"TL-{len(data.get('timeline', [])) + 1:03d}"
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    new_milestone = {
        "id": next_id,
        "date": date_str,
        "milestone": title,
        "description": description,
        "status": status
    }
    data.setdefault("timeline", []).append(new_milestone)
    save_json_to_html(TRACK_FILE, full_content, span, data)
    print(f"Added milestone {next_id} ('{title}') to track.html.")

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "--sync-git":
        sync_git()
    elif sys.argv[1] == "--add-prompt":
        if len(sys.argv) < 3:
            print("Usage: python sync_tracker.py --add-prompt \"<text>\" [source] [notes]")
            sys.exit(1)
        prompt_txt = sys.argv[2]
        src = sys.argv[3] if len(sys.argv) > 3 else "User"
        nts = sys.argv[4] if len(sys.argv) > 4 else ""
        add_prompt(prompt_txt, src, nts)
    elif sys.argv[1] == "--add-milestone":
        if len(sys.argv) < 3:
            print("Usage: python sync_tracker.py --add-milestone \"<title>\" \"<description>\" [status]")
            sys.exit(1)
        title = sys.argv[2]
        desc = sys.argv[3] if len(sys.argv) > 3 else ""
        st = sys.argv[4] if len(sys.argv) > 4 else "Pending"
        add_milestone(title, desc, st)
    else:
        print("Unknown command. Options: --sync-git, --add-prompt, --add-milestone")
