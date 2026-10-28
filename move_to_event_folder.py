"""
move_to_event_folder.py -- one-off: move results from the earlier folder layouts into the current one,
    fitted results/<id>/semi    (traces, GUI saves, fits)
    fitted results/<id>/final   (derived series, shock properties, cobpoints)
    figure/<id>                 (figures)

Run:  python move_to_event_folder.py        (prints the plan; set DRY_RUN = False to move)
Handled:
  semi fitted results/<id>/...    -> fitted results/<id>/semi/...     (every event folder of that layout)
  final fitted results/<id>/...   -> fitted results/<id>/final/...
  semi fitted results/<file>      -> fitted results/<EVENT_ID>/semi/  (the flat layout before event ids)
  final fitted results/<file>     -> fitted results/<EVENT_ID>/final/
  figure/<file>                   -> figure/<EVENT_ID>/
EVENT_ID from event_config.py. Nothing is overwritten (an item whose destination exists is skipped), and folders
left empty afterwards are removed.
"""
import re
import shutil
from pathlib import Path

import event_config as cfg

DRY_RUN = True
IS_EVENT = re.compile(r'^\d{8}-')


def plan():
    moves = []
    for old, part in (('semi fitted results', 'semi'), ('final fitted results', 'final')):
        base = Path(old)
        if not base.is_dir():
            continue
        for item in sorted(base.iterdir()):
            if item.is_dir() and IS_EVENT.match(item.name):              # an event folder of that layout
                for sub in sorted(item.iterdir()):
                    moves.append((sub, Path('fitted results') / item.name / part / sub.name))
            else:                                                        # the flat layout: this event
                moves.append((item, Path('fitted results') / cfg.EVENT_ID / part / item.name))
    fig = Path('figure')
    if fig.is_dir():
        for item in sorted(fig.iterdir()):
            if not (item.is_dir() and (item.name == cfg.EVENT_ID or IS_EVENT.match(item.name))):
                moves.append((item, fig / cfg.EVENT_ID / item.name))
    return moves


def main():
    moved = skipped = 0
    for src, dest in plan():
        if dest.exists():
            print(f'skip (exists): {src} -> {dest}')
            skipped += 1
            continue
        print(f'{"would move" if DRY_RUN else "move"}: {src} -> {dest}')
        if not DRY_RUN:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        moved += 1
    if not DRY_RUN:                                                       # tidy up emptied old folders
        for old in ('semi fitted results', 'final fitted results'):
            base = Path(old)
            if base.is_dir():
                for d in sorted(base.glob('*'), reverse=True):
                    if d.is_dir() and not any(d.iterdir()):
                        d.rmdir()
                if not any(base.iterdir()):
                    base.rmdir()
    print(f'\n{moved} item(s) {"to move" if DRY_RUN else "moved"}, {skipped} skipped'
          + ('  -- set DRY_RUN = False to move them' if DRY_RUN and moved else ''))


if __name__ == '__main__':
    main()
