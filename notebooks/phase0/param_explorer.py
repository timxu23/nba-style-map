"""
Utility — inspect any nba_api stats endpoint's parameters and result sets.

The NBA API's param names are non-obvious (0.2), and one endpoint returns
several tables. This prints, for a given endpoint:
  - every parameter name + its default (from the __init__ signature)
  - which values look filter-optional (the `_nullable` convention)
  - the result-set names, so you know which df_index to keep in fetch_or_load

Usage:
  python3 notebooks/phase0/param_explorer.py                     # default example
  python3 notebooks/phase0/param_explorer.py LeagueDashPlayerStats
  python3 notebooks/phase0/param_explorer.py ShotChartDetail
"""
import inspect
import sys

import nba_api.stats.endpoints as endpoints

# Params that are transport/plumbing, not query params — hide them.
_PLUMBING = {"self", "proxy", "headers", "timeout", "get_request"}


def describe(endpoint_name: str) -> None:
    cls = getattr(endpoints, endpoint_name, None)
    # Guard: `endpoints` also exposes lowercase SUBMODULES (e.g.
    # `commonplayerinfo`), so getattr can return a module, not an endpoint.
    # Require an actual class to avoid introspecting the wrong object.
    if not inspect.isclass(cls):
        print(f"No endpoint class named {endpoint_name!r}.")
        if cls is not None:
            print(f"  (that name resolved to a {type(cls).__name__}, not a class — "
                  f"use the CamelCase CLASS name, e.g. CommonPlayerInfo not commonplayerinfo)")
        print("List all endpoint classes:  python3 -c \"import nba_api.stats.endpoints as e; print([n for n in dir(e) if n[0].isupper()])\"")
        return

    print(f"=== {endpoint_name} — parameters (name = default) ===")
    sig = inspect.signature(cls.__init__)
    for name, p in sig.parameters.items():
        if name in _PLUMBING:
            continue
        if p.default is inspect.Parameter.empty:
            print(f"  {name:38} = <REQUIRED — no default>")
            continue
        tag = "  (optional filter)" if name.endswith("_nullable") else ""
        print(f"  {name:38} = {p.default!r}{tag}")

    # Result sets: the NAMES of the tables this endpoint returns.
    # WARNING: nba_api's static `expected_data` dict lists the right table
    # names but NOT in the runtime order of get_data_frames(). So we can show
    # you WHICH tables exist, but not a reliable df_index. Confirm the index
    # at runtime (see note printed below).
    expected = getattr(cls, "expected_data", None)
    if isinstance(expected, dict):
        print(f"\n=== result-set tables returned (names only — index NOT reliable) ===")
        for rs_name in expected.keys():
            print(f"  - {rs_name}")
        print("\n  To find the real df_index, make one call and read the live order:")
        print("    ep = <Endpoint>(**params)")
        print("    for i, rs in enumerate(ep.get_dict()['resultSets']):")
        print("        print(i, rs['name'])")
        print("  ...or just check the fetch_or_load sidecar .json — it records the")
        print("  columns of whatever df_index you pulled, so you can verify the table.")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "LeagueDashPlayerStats"
    describe(name)
