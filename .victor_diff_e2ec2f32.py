import hashlib, sqlite3, difflib
from victor.agent.prompt_section_registry import get_section_registry

r = get_section_registry()
base_map = {s.name: (s.default_text or "") for s in r.get_all()}
con = sqlite3.connect("/Users/vijaysingh/.victor/victor.db")
sections = [
    "GROUNDING_RULES_EXTENDED",
    "PARALLEL_READ_GUIDANCE",
    "LARGE_FILE_PAGINATION_GUIDANCE",
]
for sec in sections:
    base = base_map.get(sec, "")
    row = con.execute(
        "SELECT text FROM agent_prompt_candidate WHERE section_name=? AND provider='zai'",
        (sec,),
    ).fetchone()
    ev = row[0] if row else ""
    print("\n" + "=" * 70)
    print(f"SECTION: {sec}   baseline={len(base)}ch -> evolved={len(ev)}ch")
    print(
        f"baseline_md5={hashlib.md5(base.encode()).hexdigest()[:12]}  "
        f"evolved_md5={hashlib.md5(ev.encode()).hexdigest()[:12]}"
    )
    print("=" * 70)
    diff = difflib.unified_diff(
        base.splitlines(),
        ev.splitlines(),
        fromfile="baseline",
        tofile="evolved_gen1",
        lineterm="",
    )
    n = 0
    LIMIT = 45
    for line in diff:
        print(line)
        n += 1
        if n == LIMIT:
            print("...[truncated, " + str(n) + " lines shown]...")
            break
con.close()
