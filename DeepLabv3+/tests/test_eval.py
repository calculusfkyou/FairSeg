import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

from test import ensure_new


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


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
