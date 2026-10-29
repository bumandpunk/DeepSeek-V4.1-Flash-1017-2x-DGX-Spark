from spark1017.control import packet, consensus, fits_budget


def test_votes_and_state():
    p = packet(['round', [1, 17], 5], True)
    assert all(0 <= x <= 65535 for x in p)
    assert consensus([p, p], 2)
    no = p.copy(); no[4] = 0
    assert not consensus([p, no], 2)
    for bad in ([p], [p, packet(['round', [1, 18], 5], True)], [p, p[:7]], [p, p[:4]+[2,0,0,0]]):
        try: consensus(bad, 2)
        except ValueError: pass
        else: raise AssertionError('invalid consensus accepted')
    assert fits_budget(12, 80, 4, 100, 8)
    assert not fits_budget(11, 80, 4, 100, 8)
    assert not fits_budget(12, 97, 4, 100, 8)


if __name__ == '__main__':
    test_votes_and_state()
    print('control agreement, rejection and exact boundary checks passed')
