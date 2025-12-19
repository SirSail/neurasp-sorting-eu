from dataclasses import dataclass
from typing import Dict, List, Tuple
from pprint import pprint
import clingo
import sys
import os

# Adjust path to point to the correct location relative to project root
# Assuming this script is run from project root
ASP_PATH = "src/asp/sort.lp"


@dataclass(frozen=True)
class AspTest:
    name: str
    x: List[int]  # length 5


def expected_positions(x: List[int]) -> Dict[int, int]:
    """
    Stable sorting contract:
      key_i = (x_i, i) with i in 1..5
    """
    idxs = [1, 2, 3, 4, 5]
    idxs_sorted = sorted(idxs, key=lambda i: (x[i - 1], i))
    return {i: p for p, i in enumerate(idxs_sorted)}


def facts_for_x(x: List[int]) -> str:
    return " ".join(f"val({i},{x[i-1]})." for i in range(1, 6))


def solve_positions(
    asp_path: str, facts: str
) -> Tuple[int, Dict[int, int]]:
    print(f"Loading ASP file: {asp_path}")
    if not os.path.exists(asp_path):
        print(f"Error: ASP file not found at {asp_path}")
        sys.exit(1)
        
    ctl = clingo.Control(["0"])  # enumerate all models
    ctl.load(asp_path)
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])

    model_count = 0
    last_positions: Dict[int, int] = {}

    def on_model(m: clingo.Model) -> None:
        nonlocal model_count, last_positions
        model_count += 1
        tmp = {}
        for atom in m.symbols(shown=True):
            if atom.name == "pos" and len(atom.arguments) == 2:
                i = atom.arguments[0].number
                p = atom.arguments[1].number
                tmp[int(i)] = int(p)
        last_positions = tmp

    ctl.solve(on_model=on_model)
    return model_count, last_positions


def assert_is_permutation(pos: Dict[int, int]) -> None:
    assert set(pos.keys()) == {1, 2, 3, 4, 5}, f"Missing element indices: got {set(pos.keys())}"
    assert set(pos.values()) == {0, 1, 2, 3, 4}, f"Positions are not a permutation: got {set(pos.values())}"


def run_test(t: AspTest) -> None:
    print("\n" + "=" * 72)
    print(f"TEST: {t.name}")
    print("- Input x:")
    pprint(t.x)

    facts = facts_for_x(t.x)
    expected = expected_positions(t.x)

    model_count, got = solve_positions(ASP_PATH, facts)

    try:
        assert model_count == 1, f"Expected 1 model, got {model_count}"
        assert_is_permutation(got)
        assert got == expected, "Positions do not match expected ordering"

        print("- ASP facts:")
        print(f"  {facts}")

        print("- Expected positions:")
        pprint(expected, width=1)

        print("- ASP result:")
        pprint(got, width=1)

        print("RESULT: PASS")

    except AssertionError as e:
        print("RESULT: FAIL")
        print(f"Reason: {e}")

        print("\nDetails:")
        print("- ASP facts:")
        print(f"  {facts}")

        print("- Expected positions:")
        pprint(expected, width=1)

        print("- ASP result:")
        pprint(got, width=1)

        sys.exit(1)


def main() -> None:
    tests = [
        AspTest("only_positive", [8, 1, 7, 3, 5]),
        AspTest("only_negative", [-2, -9, -1, -5, -3]),
        AspTest("mixed_pos_neg", [4, -1, 0, 7, -3]),
        AspTest("mixed_with_duplicate", [2, -1, 2, 0, -1]),
        AspTest("all_zeros", [0, 0, 0, 0, 0]),
    ]

    print("Running ASP tests for sort.lp")

    for t in tests:
        run_test(t)

    print("\nAll ASP tests PASSED.")


if __name__ == "__main__":
    main()
