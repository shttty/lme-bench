# lme-bench：给 agent 的约定

命令、题集、组别见 `README.md` 和 `uv run bench.py <cmd> --help`。

## 开发循环

- 默认只跑 `bench.py dev`（DEV8）：一次 ≤ 8 题、20 分钟内出结果。48 题全量（`--set all`）是历史基线，只在维护者明确要求时跑。
- 档位：压缩（建快照）固定 luna high；Pi 组答题模型用 `--answer` 选（`ANSWERERS`，默认 `luna-high`），不进快照键，换答题模型不重建快照；判分 luna xhigh；DAG 摘要按配置档（`luna-high` 不设 `LOSSLESS_SUMMARY_THINKING`，跟会话走，即 high）。
- DAG 两组从插件仓库某个 commit `git archive` 出的 `src/`（`runs/plugin/<sha>/`）加载，不读工作区。结果写到 `runs/<arm>/<答题模型>/<profile>@<sha>/`（原生两组是 `runs/<arm>/<答题模型>/`），每行带 `answerer`、`plugin`、`profile`、`snapshot`。`dev` 跑四个 Pi 组，已答过的题跳过，所以 luna 下的原生两组直接复用历史结果。
- **别动 `SNAPSHOT_CODE` 里的函数和 `PI`**（包括注释和 docstring）：它们的源码进快照键，只为答题改它们会让全部快照重建。答题侧的东西放 `answer_env`、`answer_cmd`、`answer_one`。

## 快照缓存

- **缓存的是什么**：每题一份"3 次压缩后、第 4 段原文已追加、只差提问"的 Pi 会话 jsonl，在 `runs/pi-lcm-compacted/<key>/<题号>-<历史哈希>.jsonl`。里面有：全部 4 段原文（压缩只追加条目，原文不删，`lcm_grep` 回查的就是它）；3 条 compaction 条目（`summary` 为当时渲染进上下文的前沿，`details.nodes` 为该次新建的 DAG 节点，`firstKeptEntryId`）。Pi 原生快照另存 `runs/pi-compacted/<题号>.jsonl`（48 题都有，冻结基线，不按键）。
- **缓存键**（`snapshot_key`）：配置档名 + 以下内容的哈希：配置档环境变量、模型与档位；插件该 commit 上 `compaction.ts` 的本地 import 闭包加 `index.ts`（自动算，拆分前的 commit 退回 `index.ts` 的闭包）；`SNAPSHOT_CODE` 里那几个 bench 函数的源码（喂入、分段、压缩 RPC）；`CYCLES`、`CHARS_PER_TOKEN`、`SUMMARY_CONCURRENCY`、`API`、`PI` 启动参数；Pi SDK 版本。文件名里的历史哈希覆盖题目数据本身。以上任一变化就换键重建；只改答题侧（`tools.ts`、`search.ts`、`expand.ts`、入库侧、`answer_one` 等）的提交复用快照、只重跑答题。
- **给 bench 加新的影响压缩的函数时**，把它加进 `SNAPSHOT_CODE`；新的常量加进 `snapshot_key` 的 `bench_consts`。插件那边的分层约束在插件仓库 `AGENTS.md`。
- **`manifest.json`**：每个键目录一份，第一份快照建好时写入键的全部组成，之后每建一份记一条（插件 commit、时间、耗时）。快照结果反常时先看它。
- **不缓存**：答案与判分（按"答题模型 / profile@commit"各存一份，每个插件 commit 都重答）。答题用的 agent 目录（`models.json`、pi-lcm 的 SQLite 索引）在结果目录的 `agent/` 下，每次 `run` 开始前建好一次；索引由该 commit 的入库代码现建，不复用建快照时留下的 `runs/pi-lcm-agent/<key>/`（那里只剩建快照时的索引和压缩日志）。
- **键管不到的**：luna 服务端行为和摘要本身的随机性。同一键下快照冻结，正好让答题侧对比不受这份随机性影响；要看压缩侧改动的效果，就是换键重建后的新快照。
