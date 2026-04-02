from src.utils.stable_seed import stable_seed_offset


def test_stable_seed_offset_is_deterministic_for_strings_and_tuples():
    value_a = stable_seed_offset(("ood_large", "FULL_GA"), modulo=100000)
    value_b = stable_seed_offset(("ood_large", "FULL_GA"), modulo=100000)
    value_c = stable_seed_offset("room_alpha", modulo=100000)
    value_d = stable_seed_offset("room_alpha", modulo=100000)

    assert value_a == value_b
    assert value_c == value_d
    assert 0 <= value_a < 100000
    assert 0 <= value_c < 100000
