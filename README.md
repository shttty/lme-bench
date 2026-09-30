# lme-bench

用 LongMemEval 比较上下文压缩方法：OMP 各方法、Pi 原生压缩、`pi-lossless-context` 插件。压缩（建快照）用 `clp/gpt-6-luna`（high）；Pi 组的答题模型可选（`--answer`：`luna-high` 默认、`sol-6.1-high`），OMP 组固定 luna high；判分 `clp/gpt-6-luna`（xhigh）。OMP profile `lme-bench`：无插件、无人格提示，模型配置与日常 profile 共用。

本机设置放环境变量或本目录的 `.env`（已 gitignore）：`CLP_BASE_URL` / `CLP_API_KEY`（提供 gpt-6 系列模型的 OpenAI 兼容端点，`clp` 是它的 provider 名）、`OMP_BIN`（默认 `omp`）、`PLUGIN_REPO`（默认同级目录 `../pi-lossless-context`，Pi 组也用它 `node_modules` 里的 Pi SDK）。

## 开发时用：DEV8

```bash
uv run bench.py dev                          # 四个 Pi 组跑 DEV8（已答过的跳过），判分，出按题网格
uv run bench.py dev --answer sol-6.1-high    # 换答题模型，快照不动
uv run bench.py dev --only 778164c6          # 单题，看整条链路
uv run bench.py dev --plugin <commit>        # 指定插件版本（默认插件仓库 main 分支最新提交）
uv run bench.py report                       # 只看网格
```

- **DEV8**：8 题，都是证据只在被压缩的前 3 段里的题，类型为 single-session-user / assistant、temporal-reasoning（preference、knowledge-update、multi-session 不看）。基线 pi-native 0/8、pi-recall 6/8；pi-recall 答错的两题：15745da0 没去查，982b5123 只查到两条证据中的一条。题目内容见 `bench.py` 的 `DEV8` 注释与 `data/sample.json`。
- **插件版本固定**：DAG 组从插件仓库某个 commit 用 `git archive` 导出 `src/` 到 `runs/plugin/<sha>/` 再加载，不读工作区，跑的过程中改代码不会混进来。每行结果记 `plugin`、`profile`、`snapshot`。
- **配置档**（`PROFILES`）：`luna-high`（摘要用 luna high，插件默认参数）。
- **快照缓存**：3 次压缩后的快照按"配置档 + 插件压缩侧 + bench 喂入/压缩代码 + SDK 版本"自动算键缓存，只改答题侧的提交不重建；缓存内容、键的组成和 `manifest.json` 见 `AGENTS.md`。
- **耗时预算**：复用快照时约 3 分钟；重建时 8 题并行，每次压缩 4 路并行（共 32 路），第一次实测 23 分钟（2026-09-30）。
- 结果在 `runs/<arm>/<答题模型>/`（pi-native、pi-recall）和 `runs/<arm>/<答题模型>/<profile>@<sha>/`（DAG 组）；报告按答题模型分组。

## 历史基线：48 题

`data/sample.json` 六类各 8 题。`uv run bench.py report --set all` 看 OMP 三组和 Pi 原生两组的按类型汇总；`run`/`judge` 加 `--set all` 跑全量。

## 测法

- 每道题的 haystack 按日期排序写成真实会话 jsonl（每段会话首条用户消息前加 `[Session Date: …]`；assistant 条目带按 4.84 字符/token（luna 实测）估算的 usage，供 OMP 压缩记账）。
- **3 次压缩循环**：历史按体积切 4 段；喂第 1 段 → 压缩 → 喂第 2 段 → 压缩 → 喂第 3 段 → 压缩 → 喂第 4 段 → 提问。压缩经 RPC `compact`（handoff 用 RPC `handoff`）显式触发，OMP 组用 overlay 把 `methodOrder` 锁成单一方法，不回退。
- 组：
  - OMP：`snapcompact`、`soft`、`handoff`。`shake` 无工具输出可压，`remote` 在 clp 上不可用，均不设组。
  - Pi 原生快照（`runs/pi-compacted/`）：`pi-native` 无工具；`pi-recall` 加 9-29 recall spike 的 `history_grep` / `history_expand`（从插件仓库的 `prototype/recall-spike` 导出）。
  - 插件 DAG 快照：`pi-dag` 无工具；`pi-lcm` 加 `lcm_grep` / `lcm_expand`。插件默认 2 万 token 叶子、扇入 4、最深 3 层；不开后台预摘要，每份快照走同一条压缩路径。压缩时模型出错会退回原生摘要，`compactions[].fromHook` / `nodes` 记录每次实际用的是哪种。
- Pi 组用 Pi SDK 0.85.1、隔离 agent 目录、自动压缩关，只跑上面 3 次显式压缩；同一快照上的两组只差答题时的工具。
- 数据用 `_M`：`_S` 切段后每段约 2.7 万 token，snapcompact 在这个量级拒绝执行；`_M` 每段约 30 万 token，接近生产压缩量级。
- 答题与判分 prompt 取自官方仓库 `run_generation.py` / `evaluate_qa.py`。
- `data/` 放 HF `xiaowu0162/longmemeval` 的 `longmemeval_s.json` / `longmemeval_m.json`；`uv run bench.py sample --data m --per-type 2` 生成 `sample.json`。`run`/`judge` 可断点续跑；失败题（如 clp `content_filter`）计错并单列。
