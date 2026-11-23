import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile

from test import ensure_new, load_selection
from utils import file_sha256


def test_refuses_existing_output():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'r.json')
        ensure_new(path)                                  # does not exist yet: passes
        open(path, 'w').close()
        try:
            ensure_new(path)
        except FileExistsError:
            return
        raise AssertionError('expected FileExistsError')


def _write_selection(d, dice, es):
    ckpt = os.path.join(d, 'c.pth')
    with open(ckpt, 'wb') as f:
        f.write(b'weights')
    path = os.path.join(d, 'selection.json')
    with open(path, 'w') as f:
        json.dump({'checkpoint': {'path': ckpt, 'epoch': 3, 'sha256': file_sha256(ckpt)}, 'dice': dice, 'es': es}, f)
    return path, ckpt


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f'expected {exc.__name__}')


def test_selection_must_match_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        sel, ckpt = _write_selection(d, {'K': 2, 'channels': [5, 9]}, {'K': 7, 'channels': [1, 5, 9]})
        assert load_selection(sel, 'es', ckpt)['channels'] == [1, 5, 9]
        other = os.path.join(d, 'other.pth')
        with open(other, 'wb') as f:
            f.write(b'other weights')
        _raises(lambda: load_selection(sel, 'es', other), ValueError)


def test_selection_refuses_repeat_test_runs():
    with tempfile.TemporaryDirectory() as d:
        sel, ckpt = _write_selection(d, {'K': 0, 'channels': []}, {'K': 4, 'channels': [3]})
        _raises(lambda: load_selection(sel, 'dice', ckpt), ValueError)    # K = 0: the unmasked result is reused
        assert load_selection(sel, 'es', ckpt)['channels'] == [3]
        sel, ckpt = _write_selection(d, {'K': 4, 'channels': [3]}, {'K': 4, 'channels': [3]})
        assert load_selection(sel, 'dice', ckpt)['channels'] == [3]
        _raises(lambda: load_selection(sel, 'es', ckpt), ValueError)      # K_ES = K_Dice: the dice result is reused


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
