"""LongMemEval vs context compaction methods (OMP methods; Pi native; pi-lossless-context). Stdlib only.

History is fed in CYCLES+1 chunks with an explicit RPC compaction after each of the first CYCLES chunks, then the
question is asked. Development runs use DEV8 (8 questions whose evidence sits only in compacted chunks); the full
48-question sample is kept as the historical baseline.

  uv run bench.py dev [--profile luna-high] [--plugin REF] [--only QID]   # pi-lcm + pi-dag on DEV8, judge, report
  uv run bench.py run --arm ARM [--set dev8|all] [--profile P] [--plugin REF] [--jobs 8] [--only QID]
  uv run bench.py judge --arm ARM [--set ...] [--profile P] [--plugin REF]
  uv run bench.py report [--set dev8|all]
  uv run bench.py sample --data m --per-type 2   # stratified sample -> data/sample.json (tops up an existing one)
"""
import argparse, hashlib, inspect, json, os, random, re, shutil, subprocess, sys, threading, time, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent
# Machine-specific settings come from the environment or the gitignored .env next to this file: CLP_BASE_URL /
# CLP_API_KEY (OpenAI-compatible endpoint serving the gpt-6 models), OMP_BIN, PLUGIN_REPO.
for _line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    _k, _, _v = _line.partition("=")
    if _k.strip() and not _k.startswith("#"):
        os.environ.setdefault(_k.strip(), _v.strip())
DATA, RUNS = ROOT / "data", ROOT / "runs"
MODEL, THINKING = "gpt-6-luna", "high"
JUDGE_MODEL, JUDGE_EFFORT = "gpt-6-luna", "xhigh"
CHARS_PER_TOKEN = 4.84  # gpt-6-luna input_tokens on the largest sample chunk (1.29M chars -> 266,931)
OMP = [os.environ.get("OMP_BIN", "omp"), "--profile", "lme-bench", "--no-tools", "--no-lsp",
       "--model", f"clp/{MODEL}", f"--thinking={THINKING}", "--system-prompt", "You are a helpful assistant."]
API, KEY = os.environ.get("CLP_BASE_URL", ""), os.environ.get("CLP_API_KEY", "")
# ponytail: shake omitted (LongMemEval history has no tool results to shrink); remote omitted (clp is not a
# native OpenAI/Anthropic endpoint, OMP reports it cannot run manually).
OMP_ARMS = ("snapcompact", "soft", "handoff")
# pi-lossless-context: DAG arms load src/ exported from a commit (default: the tip of PLUGIN_BRANCH), never the
# working tree. The Pi arms run the plugin repo's own Pi SDK (its node_modules).
PLUGIN_REPO, PLUGIN_BRANCH = Path(os.environ.get("PLUGIN_REPO", ROOT.parent / "pi-lossless-context")), "main"
PI_CLI = PLUGIN_REPO / "node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
PI_FLAGS = ["--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files", "--offline"]
PI_BASE = ["node", str(PI_CLI), *PI_FLAGS]
PI_SYSTEM = ["--system-prompt", "You are a helpful assistant."]
PI = [*PI_BASE, "--model", f"clp/{MODEL}:{THINKING}", *PI_SYSTEM]  # snapshot builds (compaction) always run luna high
# The model that answers from a snapshot (--answer). Not part of the snapshot key: every answer model reads the same
# snapshots. gpt-6.1-sol reuses luna's models.json entry (372k context, as OMP declares for the gpt-6 family).
ANSWERERS = {"luna-high": ("gpt-6-luna", "high"), "sol-6.1-high": ("gpt-6.1-sol", "high")}
# pi-native / pi-recall answer from one Pi native-compaction snapshot per question (no plugin); pi-recall adds the
# 9-29 recall spike's history_grep/history_expand. pi-dag / pi-lcm answer from the plugin's DAG snapshot for a
# --profile at a pinned --plugin commit; pi-lcm adds lcm_grep/lcm_expand.
NATIVE_ARMS = ("pi-native", "pi-recall")
DAG_ARMS = ("pi-dag", "pi-lcm")
ARMS = (*OMP_ARMS, *NATIVE_ARMS, *DAG_ARMS)
CYCLES = 3

# 8 questions (single-session-user/assistant, temporal-reasoning) whose evidence is only in the 3 compacted chunks.
# pi-native 0/8; pi-recall 6/8 (15745da0: never searched; 982b5123: found one of the two facts).
DEV8 = ("778164c6", "51b23612", "ceb54acb", "577d4d32", "3d86fd0a", "15745da0", "gpt4_65aabe59", "982b5123")

# Summary profiles = plugin env for the compaction.
PROFILES = {"luna-high": {}}
# A DAG snapshot is keyed (snapshot_key) by everything that decides what its compactions produce: the profile, the
# plugin's compaction side at the pinned commit, the bench code that feeds and compacts the history, the Pi SDK version.
# A plugin commit that only changes the answer side (tools, search, capture) reuses the snapshots.
LOCAL_IMPORT = re.compile(r'from "\./([\w.-]+\.ts)"')
SUMMARY_CONCURRENCY = "4"  # per compaction; with --jobs 8 that is 32 parallel summary calls (the level measured on clp)

# Official LongMemEval prompts (src/generation/run_generation.py, src/evaluation/evaluate_qa.py).
ASK = ("Please answer the question based on the relevant chat history above. Answer the question step by step: "
       "first extract all the relevant information, and then reason over the information to get the answer.\n\n"
       "Current Date: {}\nQuestion: {}\nAnswer (step by step):")
_BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
         "contains the correct answer. Otherwise, answer no. ")
_STEPS = ("If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct "
          "answer, you should also answer yes. If the response only contains a subset of the information required by the "
          "answer, answer no. ")
_TAIL = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE = {
    "single-session-user": _BASE + _STEPS + _TAIL,
    "single-session-assistant": _BASE + _STEPS + _TAIL,
    "multi-session": _BASE + _STEPS + _TAIL,
    "temporal-reasoning": _BASE + _STEPS + "In addition, do not penalize off-by-one errors for the number of days. If the "
        "question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting "
        "19 days when the answer is 18), the model's response is still correct. " + _TAIL,
    "knowledge-update": _BASE + "If the response contains some previous information along with an updated answer, the "
        "response should be considered as correct as long as the updated answer is the required answer." + _TAIL,
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model "
        "does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes "
        "the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model "
        "response correct? Answer yes or no only.",
}
ABSTAIN = ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the "
           "model correctly identifies the question as unanswerable. The model could say that the information is incomplete, "
           "or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\n"
           "Model Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.")


def parse_date(s):  # "2023/05/20 (Sat) 02:21"
    return datetime.strptime(re.sub(r" \(\w+\)", "", s), "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_session(q, path, pi=False):
    """Haystack -> v3 session jsonl (OMP: title slot + header; Pi: header), linear message chain."""
    sessions = sorted(((parse_date(d), d, s) for d, s in zip(q["haystack_dates"], q["haystack_sessions"]) if s),
                      key=lambda x: x[0])
    start = sessions[0][0]
    title = json.dumps({"type": "title", "v": 1, "title": q["question_id"], "updatedAt": iso(start), "pad": ""},
                       separators=(",", ":"))
    lines = [] if pi else [title.replace('"pad":""', '"pad":"' + " " * (255 - len(title.encode())) + '"')]
    lines.append(json.dumps({"type": "session", "version": 3, "id": str(uuid.uuid4()), "timestamp": iso(start),
                             "cwd": str(RUNS / "cwd")}))
    parent, n, chars = None, 0, 0
    for when, label, turns in sessions:
        turns = [dict(t) for t in turns]
        if turns[0]["role"] == "user":
            turns[0]["content"] = f"[Session Date: {label}]\n{turns[0]['content']}"
        else:
            turns.insert(0, {"role": "user", "content": f"[Session Date: {label}]"})
        for t in turns:
            ts = when + timedelta(seconds=n)
            n += 1
            eid = f"{n:08x}"
            ms = int(ts.timestamp() * 1000)
            content = [{"type": "text", "text": t["content"]}]
            chars += len(t["content"])
            if t["role"] == "user":
                msg = {"role": "user", "content": content, "attribution": "user", "timestamp": ms}
            else:
                # ponytail: estimated usage for OMP's context accounting; a zero here makes snapcompact
                # refuse to run.
                est = int(chars / CHARS_PER_TOKEN)
                msg = {"role": "assistant", "content": content, "api": "openai-responses", "provider": "clp",
                       "model": MODEL, "usage": {"input": est, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                       "totalTokens": est, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
                       "stopReason": "stop", "timestamp": ms}
            lines.append(json.dumps({"type": "message", "id": eid, "parentId": parent, "timestamp": iso(ts),
                                     "message": msg}, ensure_ascii=False))
            parent = eid
    assert pi or len(lines[0].encode()) == 255, len(lines[0].encode())
    path.write_text("\n".join(lines) + "\n")


def load_sample():
    return json.loads((DATA / "sample.json").read_text())


def cmd_sample(a):
    qs = json.loads((DATA / f"longmemeval_{a.data}.json").read_text())
    prev = json.loads((DATA / "sample.json").read_text()) if (DATA / "sample.json").exists() else []
    rng, out = random.Random(a.seed), []
    for t in sorted({q["question_type"] for q in qs}):
        byid = {q["question_id"]: q for q in qs if q["question_type"] == t}
        keep = [byid[q["question_id"]] for q in prev if q["question_id"] in byid][:a.per_type]
        ids = {q["question_id"] for q in keep}
        pool = [q for q in qs if q["question_type"] == t and q["question_id"] not in ids]
        out += keep + rng.sample(pool, min(a.per_type - len(keep), len(pool)))
    (DATA / "sample.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"{len(out)} questions ->", DATA / "sample.json")


def chunk_cuts(msgs, parts):
    """Indexes splitting msgs into `parts` roughly equal-size chunks, each starting at a user message."""
    sizes = [len(m) for m in msgs]
    total, cuts, acc, k = sum(sizes), [], 0, 1
    for i, s in enumerate(sizes):
        if k < parts and acc >= total * k / parts and json.loads(msgs[i])["message"]["role"] == "user":
            cuts.append(i); k += 1
        acc += s
    return cuts


def jsonl_lines(path):
    """JSONL rows split on '\\n' only; str.splitlines() also breaks on U+2028 etc. inside ensure_ascii=False JSON."""
    return [l for l in path.read_text().split("\n") if l]


def append_entries(sess, chunk):
    """Append raw history after the current leaf. Assistant usage is re-estimated from the latest
    compaction's tokensAfter so OMP's context accounting matches what is actually live."""
    rows = [json.loads(l) for l in jsonl_lines(sess)]
    parent = rows[-1]["id"]
    base = next((r.get("tokensAfter") or 0 for r in reversed(rows) if r.get("type") == "compaction"), 0)
    out, chars = [], 0
    for line in chunk:
        o = json.loads(line); o["parentId"] = parent; parent = o["id"]
        m = o["message"]
        chars += sum(len(b.get("text", "")) for b in m["content"])
        if m["role"] == "assistant":
            m["usage"]["input"] = m["usage"]["totalTokens"] = base + int(chars / CHARS_PER_TOKEN)
        out.append(json.dumps(o, ensure_ascii=False))
    with sess.open("a") as f:
        f.write("\n".join(out) + "\n")


def overlay(arm):
    """Per-arm config overlay forcing a single compaction method (no fallback). Auto compaction is off so
    the only compactions are the CYCLES explicit RPC ones."""
    p = RUNS / arm / "overlay.yml"
    p.write_text(f"compaction:\n  enabled: false\n  methodOrder: [{arm}]\n")
    return ["--config", str(p)]


def export_tree(ref, path):
    """`path` of pi-lossless-context at commit `ref`, exported once to runs/plugin/<sha>/ (git archive, never the
    working tree, so edits during a run cannot leak in). Returns (short sha, exported path)."""
    git = ["git", "-C", str(PLUGIN_REPO)]
    sha = subprocess.run([*git, "rev-parse", "--short=10", ref], capture_output=True, text=True, check=True).stdout.strip()
    out = RUNS / "plugin" / sha
    if not (out / path).exists():
        out.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run([*git, "archive", sha, path], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(out)], input=tar, check=True)
    return sha, out / path


class Pin(NamedTuple):
    """Plugin build + summary profile of a DAG run."""
    sha: str
    ext: Path
    profile: str
    snapshot: str  # snapshot_key(); runs with the same key share DAG snapshots
    parts: dict  # what the key hashes, written to the snapshot dir's manifest.json

    @property
    def tag(self):
        return f"{self.profile}@{self.sha}"


def pin_plugin(a):
    sha, src = export_tree(a.plugin, "src")
    return Pin(sha, src / "index.ts", a.profile, *snapshot_key(src, a.profile))


PI_ARMS = (*NATIVE_ARMS, *DAG_ARMS)


def answer_args(arm, pin):
    if arm == "pi-recall":
        _, spike = export_tree(PLUGIN_BRANCH, "prototype/recall-spike")
        return ["-e", str(spike / "recall-extension.ts"), "--tools", "history_grep,history_expand"]
    if arm == "pi-lcm":
        return ["-e", str(pin.ext), "--tools", "lcm_grep,lcm_expand"]
    return ["--no-tools"]


def results_dir(arm, pin, answer):
    """runs/<arm>/ for the OMP arms (they always answer with OMP's luna); runs/<arm>/<answer>/[<profile>@<sha>/] for Pi."""
    if arm in OMP_ARMS:
        return RUNS / arm
    return RUNS / arm / answer / pin.tag if pin else RUNS / arm / answer


def answer_env(pin, answer, agent):
    """Pi agent dir for answering: pi_env's, plus the answer model in models.json. Set up once per run before the
    parallel answers start, so no Pi process reads a models.json that another thread is rewriting."""
    env = pi_env(pin, agent)
    model, _ = ANSWERERS[answer]
    if model != MODEL:
        p = agent / "models.json"
        m = json.loads(p.read_text())
        clp = m["providers"]["clp"]
        clp["models"].append({**clp["models"][0], "id": model, "name": model})
        p.write_text(json.dumps(m))
    return env


def answer_cmd(answer):
    model, thinking = ANSWERERS[answer]
    return [*PI_BASE, "--model", f"clp/{model}:{thinking}", *PI_SYSTEM]


def pi_env(pin, agent):
    """Pi agent dir `agent`: clp luna only, auto compaction off (only the staged manual compactions run). With the
    plugin loaded it also holds the plugin's lossless-context.db/.log."""
    agent.mkdir(parents=True, exist_ok=True)
    (agent / "models.json").write_text(json.dumps({"providers": {"clp": {"baseUrl": API, "api": "openai-responses",
        "apiKey": KEY, "models": [{"id": MODEL, "name": MODEL, "reasoning": True, "input": ["text", "image"],
        "contextWindow": 372000, "maxTokens": 128000, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}))
    (agent / "settings.json").write_text(json.dumps({"compaction": {"enabled": False}}))
    env = {**os.environ, "PI_CODING_AGENT_DIR": str(agent)}
    if pin:
        # plugin defaults otherwise (20k-token leaves, fan-in 4, depth 3); background pre-summarization off, so every
        # snapshot takes the same compaction path
        env.update({"LOSSLESS_SUMMARY_CONCURRENCY": SUMMARY_CONCURRENCY, "LOSSLESS_PRESUMMARIZE_AT": "1",
                    **PROFILES[pin.profile]})
    return env


def compact_once(arm, sess, pin):
    """One RPC compaction on the session file; returns an error string or None."""
    if arm in PI_ARMS:
        ext = ["-e", str(pin.ext)] if pin else []
        env = pi_env(pin, RUNS / "pi-lcm-agent" / pin.snapshot if pin else RUNS / "pi-agent")
        cmd, req = PI + ext + ["--no-tools", "--mode", "rpc", "--session", str(sess)], '{"id":"c","type":"compact"}'
    else:
        cmd = OMP + overlay(arm) + ["--mode", "rpc", "--no-ui", "--session-dir", str(sess.parent), "--resume", str(sess)]
        env, req = None, f'{{"id":"c","type":"{"handoff" if arm == "handoff" else "compact"}"}}'
    errf = sess.with_name(sess.name + ".compact.err")
    with errf.open("w") as ef:
        p = subprocess.Popen(cmd, cwd=RUNS / "cwd", env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=ef, text=True)
        p.stdin.write(req + "\n"); p.stdin.flush()
        resp = None
        for line in p.stdout:  # keep stdin open until the response arrives; EOF may end the process early
            if line.startswith("{") and '"id":"c"' in line.replace(" ", "") and '"response"' in line:
                resp = json.loads(line); break
        p.stdin.close()
        p.wait(timeout=120)
    if not (resp and resp.get("success")):
        return (resp or {}).get("error") or errf.read_text()[-500:]
    return None


def staged_compaction(arm, sess, pin):
    """Feed history chunk by chunk, compacting via RPC after each of the first CYCLES chunks."""
    lines = jsonl_lines(sess)
    nh = 1 if arm in PI_ARMS else 2
    head, msgs = lines[:nh], lines[nh:]
    cuts = chunk_cuts(msgs, CYCLES + 1)
    bounds = [0] + cuts + [len(msgs)]
    sess.write_text("\n".join(head + msgs[:bounds[1]]) + "\n")
    for i in range(CYCLES):
        if i:
            append_entries(sess, msgs[bounds[i]:bounds[i + 1]])
        err = compact_once(arm, sess, pin)
        if err:
            return f"compaction {i + 1} failed: {err}"
    append_entries(sess, msgs[bounds[CYCLES]:])
    return None


# The bench code that feeds and compacts the history: part of the DAG snapshot key.
SNAPSHOT_CODE = (build_session, chunk_cuts, jsonl_lines, append_entries, pi_env, compact_once, staged_compaction)
MANIFEST_LOCK = threading.Lock()


def digest(b):
    return hashlib.sha1(b).hexdigest()[:10]


def compaction_files(src):
    """Plugin files deciding what a compaction produces, by content hash: the local import closure of compaction.ts
    (of index.ts in commits before the split), plus index.ts, which wires the hooks."""
    files, todo = set(), ["compaction.ts" if (src / "compaction.ts").exists() else "index.ts"]
    while todo:
        f = todo.pop()
        if f not in files:
            files.add(f)
            todo += LOCAL_IMPORT.findall((src / f).read_text())
    return {f: digest((src / f).read_bytes()) for f in sorted(files | {"index.ts"})}


def snapshot_key(src, profile):
    """(key, parts) of a DAG snapshot. parts is everything that decides what the staged compactions produce, the summary
    model's own randomness aside; the key is the profile + a hash of parts."""
    parts = {"profile": profile, "profile_env": PROFILES[profile], "model": f"clp/{MODEL}:{THINKING}",
             "plugin_files": compaction_files(src),
             "bench_code": digest("".join(inspect.getsource(f) for f in SNAPSHOT_CODE).encode()),
             "bench_consts": {"CYCLES": CYCLES, "CHARS_PER_TOKEN": CHARS_PER_TOKEN,
                              "SUMMARY_CONCURRENCY": SUMMARY_CONCURRENCY, "API": API, "PI": PI[2:]},
             "sdk": json.loads((PI_CLI.parents[2] / "package.json").read_text())["version"]}  # PI[2:]: no local path
    return f"{profile}-{digest(json.dumps(parts, sort_keys=True).encode())}", parts


def history_hash(q):
    """The question's raw history (what build_session reads): part of each DAG snapshot's file name."""
    return digest(json.dumps([q["haystack_dates"], q["haystack_sessions"]], sort_keys=True).encode())


def record_build(pin, name, seconds):
    """Log a finished snapshot in <key>/manifest.json (created with the key's parts by the first build), so an odd
    snapshot can be traced to the plugin commit, SDK and model that built it."""
    with MANIFEST_LOCK:
        p = RUNS / "pi-lcm-compacted" / pin.snapshot / "manifest.json"
        m = json.loads(p.read_text()) if p.exists() else {"key": pin.snapshot, **pin.parts, "builds": {}}
        m["builds"][name] = {"plugin": pin.sha, "at": iso(datetime.now(timezone.utc)), "seconds": seconds}
        p.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n")


def answer_one(arm, q, pin, answer, env):
    d = results_dir(arm, pin, answer)
    sess = d / "sessions" / f"{q['question_id']}.jsonl"
    t0 = time.time()
    meta = {"question_id": q["question_id"], "question_type": q["question_type"], "arm": arm,
            **({"answerer": answer} if arm in PI_ARMS else {}),
            **({"plugin": pin.sha, "profile": pin.profile, "snapshot": pin.snapshot} if pin else {})}
    fail = {**meta, "rc": -1, "answer": "", "compactions": []}
    built = None
    if arm in PI_ARMS:
        # DAG snapshots: one per question history under the snapshot key. The native snapshots (runs/pi-compacted)
        # are the frozen 48-question baseline and are not keyed.
        snap = (RUNS / "pi-lcm-compacted" / pin.snapshot / f"{q['question_id']}-{history_hash(q)}.jsonl" if pin
                else RUNS / "pi-compacted" / f"{q['question_id']}.jsonl")
        if not snap.exists():
            snap.parent.mkdir(parents=True, exist_ok=True)
            tmp = snap.with_suffix(f".{arm}.tmp")
            build_session(q, tmp, pi=True)
            err = staged_compaction(arm, tmp, pin)
            if err:
                return {**fail, "seconds": round(time.time() - t0, 1), "stderr": err}
            tmp.rename(snap)
            built = round(time.time() - t0, 1)
            if pin:
                record_build(pin, snap.name, built)
        shutil.copyfile(snap, sess)
        # answers run in the results dir's own agent dir; for pi-lcm that means an index built by the answer-side code
        # of this commit, never the one left over from the snapshot build
        cmd = answer_cmd(answer) + answer_args(arm, pin) + ["--session", str(sess)]
    else:
        build_session(q, sess)
        err = staged_compaction(arm, sess, None)
        if err:
            return {**fail, "seconds": round(time.time() - t0, 1), "stderr": err}
        cmd, env = OMP + overlay(arm) + ["--session-dir", str(d / "sessions"), "--resume", str(sess)], None
    prompt = ASK.format(q["question_date"], q["question"])
    p = subprocess.run(cmd + ["-p", prompt], cwd=RUNS / "cwd", env=env, capture_output=True, text=True, timeout=1800)
    entries = [json.loads(l) for l in jsonl_lines(sess)[1:]]
    comps = [{**{k: e.get(k) for k in ("method", "tokensBefore", "tokensAfter", "fromHook")}, "summaryChars": len(e.get("summary") or ""),
              "nodes": len(e["details"].get("nodes") or []) if isinstance(e.get("details"), dict) else 0}
             for e in entries if e.get("type") == "compaction"]
    tools = [e["message"].get("toolName") for e in entries if (e.get("message") or {}).get("role") == "toolResult"]
    return {**meta, "rc": p.returncode, "seconds": round(time.time() - t0, 1), "snapshot_seconds": built,
            "answer": p.stdout.strip(), "stderr": p.stderr[-2000:], "compactions": comps, "tool_calls": tools}


def done(path, key="question_id"):
    return {json.loads(l)[key] for l in jsonl_lines(path)} if path.exists() else set()


def select(a):
    ids = set(DEV8) if a.set == "dev8" else None
    return [q for q in load_sample() if (ids is None or q["question_id"] in ids) and (not a.only or q["question_id"] == a.only)]


def cmd_run(a):
    if a.arm in OMP_ARMS and a.answer != "luna-high":
        sys.exit(f"{a.arm} always answers with OMP's {MODEL} {THINKING}; --answer applies to the Pi arms")
    pin = pin_plugin(a) if a.arm in DAG_ARMS else None
    d = results_dir(a.arm, pin, a.answer)
    (d / "sessions").mkdir(parents=True, exist_ok=True)
    (RUNS / "cwd").mkdir(parents=True, exist_ok=True)
    env = answer_env(pin, a.answer, d / "agent") if a.arm in PI_ARMS else None
    out = d / "answers.jsonl"
    qs = [q for q in select(a) if q["question_id"] not in done(out)]
    with ThreadPoolExecutor(a.jobs) as pool, out.open("a") as f:
        for r in pool.map(lambda q: answer_one(a.arm, q, pin, a.answer, env), qs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            snap = f" (snapshot built in {r['snapshot_seconds']}s)" if r.get("snapshot_seconds") else ""
            print(r["question_id"], r["arm"], r.get("answerer", ""), pin.tag if pin else "", "rc", r["rc"],
                  f"{r['seconds']}s{snap}", "tools", len(r.get("tool_calls") or []), flush=True)


def chat(prompt):
    body = json.dumps({"model": JUDGE_MODEL, "reasoning_effort": JUDGE_EFFORT, "max_tokens": 8000,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{API}/chat/completions", body, {"Authorization": f"Bearer {KEY}",
                                 "Content-Type": "application/json", "User-Agent": "curl/8.9.1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def judge_one(q, r):
    tpl = ABSTAIN if q["question_id"].endswith("_abs") else JUDGE[q["question_type"]]
    verdict = chat(tpl.format(q["question"], q["answer"], r["answer"]))
    return {**r, "verdict": verdict, "correct": "yes" in verdict.lower() and r["rc"] == 0}


def cmd_judge(a):
    d = results_dir(a.arm, pin_plugin(a) if a.arm in DAG_ARMS else None, a.answer)
    qs = {q["question_id"]: q for q in load_sample()}
    out = d / "judged.jsonl"
    todo = [r for r in map(json.loads, jsonl_lines(d / "answers.jsonl"))
            if r["question_id"] not in done(out)]
    with ThreadPoolExecutor(a.jobs) as pool, out.open("a") as f:
        for r in pool.map(lambda r: judge_one(qs[r["question_id"]], r), todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()


def rows_of(path):
    return {r["question_id"]: r for r in map(json.loads, jsonl_lines(path))} if path.exists() else {}


def report_all():
    """The historical 48-question table (6 types) for the OMP and Pi-native arms (Pi arms answered by luna high)."""
    arms = (*OMP_ARMS, *NATIVE_ARMS)
    rows, errs = {}, {}
    for arm in arms:
        for r in rows_of(results_dir(arm, None, "luna-high") / "judged.jsonl").values():
            rows.setdefault(r["question_type"], {}).setdefault(arm, []).append(r["correct"])
            if r["rc"] != 0:
                errs.setdefault(arm, []).append(f"{r['question_id']}: {r['stderr'].strip().splitlines()[-1][:120]}")
    print(f"{'type':28} " + " ".join(f"{a:>12}" for a in arms))
    tot = {a: [] for a in arms}
    for t in sorted(rows):
        cells = []
        for a in arms:
            v = rows[t].get(a, []); tot[a] += v
            cells.append(f"{sum(v)}/{len(v)}".rjust(12))
        print(f"{t:28} " + " ".join(cells))
    print(f"{'TOTAL':28} " + " ".join((f"{sum(v)}/{len(v)} {sum(v)/len(v):.0%}" if v else "-").rjust(12) for v in tot.values()))
    for arm, lines in errs.items():
        print(f"\n{arm} failures (counted wrong):", *lines, sep="\n  ")


def report_dev():
    """DEV8 grid: one row per question, one column per answer model and arm (DAG arms per profile@commit, oldest
    first), grouped by answer model."""
    cols = []
    for answer in ANSWERERS:
        cols += [(f"{answer}: {arm}", results_dir(arm, None, answer)) for arm in NATIVE_ARMS
                 if results_dir(arm, None, answer).exists()]
        for arm in DAG_ARMS:
            cols += [(f"{answer}: {arm} {d.name}", d) for d in sorted((RUNS / arm / answer).glob("*@*"),
                                                                        key=lambda p: p.stat().st_mtime)]
    types = {q["question_id"]: q["question_type"].replace("single-session-", "ss-") for q in load_sample()}
    judged = [rows_of(d / "judged.jsonl") for _, d in cols]
    answered = [rows_of(d / "answers.jsonl") for _, d in cols]
    for i, (name, _) in enumerate(cols):
        ans = [answered[i][q] for q in DEV8 if q in answered[i]]
        built = [r["snapshot_seconds"] for r in ans if r.get("snapshot_seconds")]
        timing = (f"; answered {len(ans)}, slowest {max(r['seconds'] for r in ans):.0f}s" if ans else "") + \
                 (f", snapshots built {len(built)} (slowest {max(built):.0f}s)" if built else "")
        print(f"C{i + 1} = {name}{timing}")
    print(f"\n{'question':16}{'type':22}" + "".join(f"C{i + 1:<6}" for i in range(len(cols))))
    tot = [[0, 0] for _ in cols]
    for qid in DEV8:
        cells = []
        for i in range(len(cols)):
            r = judged[i].get(qid)
            if r:
                tools = len(r.get("tool_calls") or [])
                cells.append(("Y" if r["correct"] else "x") + (str(tools) if tools else ""))
                tot[i][0] += r["correct"]; tot[i][1] += 1
            else:
                cells.append("?" if qid in answered[i] else ".")
        print(f"{qid:16}{types[qid]:22}" + "".join(f"{c:7}" for c in cells))
    print(f"{'TOTAL':38}" + "".join(f"{f'{c}/{n}':7}" for c, n in tot))
    print("\nY right, x wrong, digit = tool calls, ? answered not judged, . not run")


def cmd_report(a):
    report_all() if a.set == "all" else report_dev()


def cmd_dev(a):
    """DEV8 loop for one answer model: pi-lcm first (builds any missing DAG snapshots), then pi-dag on the same
    snapshots, then pi-native / pi-recall on the frozen native snapshots (skipped where already answered); judge;
    report."""
    a.set = "dev8"
    pin = pin_plugin(a)
    t0 = time.time()
    print(f"answer {a.answer}, plugin {pin.sha}, profile {pin.profile}, snapshot key {pin.snapshot}", flush=True)
    arms = ("pi-lcm", "pi-dag", *NATIVE_ARMS)
    for arm in arms:
        a.arm = arm
        cmd_run(a)
    for arm in arms:
        a.arm = arm
        cmd_judge(a)
    report_dev()
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def plugin_args(s):
        s.add_argument("--profile", choices=PROFILES, default="luna-high")
        s.add_argument("--plugin", default=PLUGIN_BRANCH, help="pi-lossless-context commit/branch for DAG arms")
        s.add_argument("--only", help="one question id")
        s.add_argument("--jobs", type=int, default=8)
        s.add_argument("--answer", choices=ANSWERERS, default="luna-high", help="answer model of the Pi arms")

    s = sub.add_parser("sample"); s.add_argument("--data", choices=("s", "m"), default="m"); s.add_argument("--per-type", type=int, default=2); s.add_argument("--seed", type=int, default=0)
    for name in ("run", "judge"):
        s = sub.add_parser(name); s.add_argument("--arm", choices=ARMS, required=True)
        s.add_argument("--set", choices=("dev8", "all"), default="dev8"); plugin_args(s)
    plugin_args(sub.add_parser("dev"))
    s = sub.add_parser("report"); s.add_argument("--set", choices=("dev8", "all"), default="dev8")
    a = ap.parse_args()
    {"sample": cmd_sample, "run": cmd_run, "judge": cmd_judge, "report": cmd_report, "dev": cmd_dev}[a.cmd](a)
