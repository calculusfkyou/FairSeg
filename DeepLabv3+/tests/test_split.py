import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from make_split import split_report, stratified_split


def test_exact_quotas():
    names = [f'f{i}' for i in range(80)]
    keys = ['a'] * 40 + ['b'] * 24 + ['c'] * 16
    train, val = stratified_split(names, keys, n_val=10, seed=42)
    assert len(train) == 70 and len(val) == 10 and not set(train) & set(val)
    assert set(train) | set(val) == set(names)
    by_key = Counter(keys[names.index(n)] for n in val)
    assert by_key == {'a': 5, 'b': 3, 'c': 2}
    assert stratified_split(names, keys, 10, 42) == (train, val)


def test_largest_remainder():
    names = [f'f{i}' for i in range(20)]
    keys = ['a'] * 7 + ['b'] * 7 + ['c'] * 6        # 名額 1.75／1.75／1.5
    _, val = stratified_split(names, keys, n_val=5, seed=42)
    assert Counter(keys[names.index(n)] for n in val) == {'a': 2, 'b': 2, 'c': 1}


def test_report_excludes_unknown():
    attrs = {'n1': {'ethnicity': 1}, 'n2': {'ethnicity': 0}, 'n3': {'ethnicity': -1}, 'n4': {'ethnicity': 0}}
    rows = {r['group']: r for r in split_report(attrs, ['n1', 'n2', 'n3'], ['n4'], attr_names=('ethnicity',))}
    assert rows['Hispanic']['n_train'] == 1 and rows['Hispanic']['pct_train'] == 50.0
    assert rows['Hispanic']['pct_val'] == 0.0
    assert rows['Unknown']['pct_train'] is None


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
