"""Нативный диалог выбора папки через tkinter.

Streamlit запускается как локальный сервер, поэтому Tk-диалог открывается на
машине пользователя — это даёт привычный системный проводник вместо ручного
ввода пути в text_input. На headless-серверах функция вернёт None и UI должен
fallback'нуться на text_input.
"""

from __future__ import annotations

from pathlib import Path


def pick_directory(title: str = "Выберите папку") -> Path | None:
    """Открывает системный диалог выбора папки. Блокирует поток до закрытия.

    Возвращает None, если пользователь отменил выбор или Tk недоступен (Linux
    без $DISPLAY, Streamlit Cloud и пр.).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    try:
        root = tk.Tk()
    except tk.TclError:
        return None

    root.withdraw()
    root.wm_attributes("-topmost", True)
    try:
        chosen = filedialog.askdirectory(title=title, parent=root)
    finally:
        root.destroy()

    if not chosen:
        return None
    return Path(chosen)


def is_picker_available() -> bool:
    """Быстрая проверка, что Tk доступен — без открытия диалога."""
    try:
        import tkinter as tk
    except ImportError:
        return False
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False
