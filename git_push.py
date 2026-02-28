#!/usr/bin/env python3
"""
Đẩy thay đổi lên git: add, commit, push.
Chạy: python git_push.py [message]
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path = PROJECT_DIR) -> bool:
    """Chạy lệnh, trả True nếu thành công."""
    r = subprocess.run(cmd, cwd=str(cwd), shell=False)
    return r.returncode == 0


def main():
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        message = "Update portfolio allocation, risk optimizer, and export logic"

    print("=" * 60)
    print(" GIT PUSH")
    print("=" * 60)

    if not run(["git", "add", "-A"]):
        print("Lỗi: git add thất bại")
        sys.exit(1)
    print("✓ git add -A")

    if not run(["git", "commit", "-m", message]):
        print("Lỗi: git commit thất bại (có thể không có thay đổi)")
        sys.exit(1)
    print(f"✓ git commit -m \"{message}\"")

    if not run(["git", "push"]):
        print("Lỗi: git push thất bại")
        sys.exit(1)
    print("✓ git push")

    print("=" * 60)
    print(" Đã đẩy lên git thành công")
    print("=" * 60)


if __name__ == "__main__":
    main()
