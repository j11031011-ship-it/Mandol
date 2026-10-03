# Mandol 代码阅读文档

> 依据《程序代码阅读理解导引.md》的方法编写，是本仓库的系统化阅读手册。
> **阅读目标**：从"能跑通 README 示例"到"能画出架构、讲清三条核心流程、指出已知缺陷"。
> **伴侣文档**：《系统分析与设计报告.md》（任务背景、需求与方案）、《优化方向.md》（融合索引增强方向深挖）。
> 行号以当前工作区版本为准（所有引用均经源码核验）。

## 阅读路线总览

```
第 0 步  跑通 README 快速示例（可选；建立观察窗口，CPU 可跑）
第 1 步  粗读：根目录 → pyproject → src/mandol 包结构 → 入口与模块地图（§一）
第 2 步  精读 core/：MemoryUnit → SemanticMap（写入/过滤/换页）→ SemanticGraph（§二）
第 3 步  精读 retrieval/：接口 → smart_search 主链路 → 三路后端 → 融合/重排（§二）
第 4 步  精读 storage/：RocksDB 分层换页（与 core 换出钩子成对读）
第 5 步  按需深入：triple_retrieval / quantification / auto_builder / 三塔
第 6 步  产出：架构图 + 模块说明卡 + 问题记录表（§四）
```

时间分配建议：粗读 20%，core 30%，retrieval 30%，其余 20%。

---

## 一、粗读阶段

### 1.1 程序整体结构（四步法应用）

**步骤 1：顶层目录"形状"（角色标签）**


| 目录 / 文件                                                         | 角色               | 说明                                                                                               |
| --------------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------- |
| `src/mandol/`                                                       | 源码主体           | 13 个子包（见 1.2）                                                                                |
| `tests/`                                                            | 核心测试           | 仅`test_rocksdb_tiered_cache.py` 一个文件                                                          |
| `examples/mandol_chat/`                                             | 可运行 Web 示例    | FastAPI + 前端静态页 + 独立测试套件（它的测试是 Makefile 的`TEST_DIR`）                            |
| `benchmark_locomo/` `benchmark_longmemeval/` `benchmark_self_host/` | 评测资产           | 各含`dataset/ dataset_maker/ scripts/ task_eval/` 与 `REPRODUCE.md`                                |
| `docs/`                                                             | Sphinx 文档（rst） | `docs/current/` 为现行文档（quickstart、data-structures、retrieval、persistence、configuration…） |
| `website/`                                                          | Docusaurus 官网    | TS 前端，与阅读主线无关                                                                            |
| `release-notes/` `picture/` `README.assets/`                        | 资料               | 发布说明与图片资产                                                                                 |
| `pyproject.toml` + `uv.lock`                                        | 依赖与锁定         | 项目用**uv** 管理；Python 3.12                                                                     |
| `Makefile`                                                          | 任务入口           | test / lint / syntax / docs / build（见 1.3）                                                      |
| `env.template`                                                      | 环境变量模板       | LLM 凭据、HF、重排后端、调优开关（见 1.3）                                                         |
| `.pre-commit-config.yaml` `vllm.sh`                                 | 工具链             | 提交钩子；vLLM 服务脚本                                                                            |

**步骤 2：划分方式判定。** `src/mandol` 是"垂直子系统 + 内部分层"的混合结构：

- 模型与存储层：`core/`（`SemanticMap`、`SemanticGraph`、`MemoryUnit`、`MemorySpace`）
- 检索管线：`retrieval/`（三路后端 + 融合 + 重排 + 图检索）
- 高阶编排：`triple_retrieval/`（三塔）、`memory_router/`（路由）、`quantification/`（量化）、
  `hierarchical|episodic|entity_relation/`（三塔各自的检索器）
- 生命周期：`storage/`（RocksDB 换页）、`auto_builder/`（高阶记忆构建）
- 基础能力：`llm/`、`cluster/`、`utils/`

→ 追一条查询要横穿：`core`（存储）→ `retrieval`（召回）→ `triple_retrieval`（编排）。

**步骤 3：核心文件候选名单（Top 10，体量为实测）**


| 文件                                         | 体量   | 为什么先读                                          |
| ---------------------------------------------- | -------- | ----------------------------------------------------- |
| `core/semantic_map.py`                       | 136 KB | 融合存储"上帝类"：写入 / 索引 / 过滤 / 换页全在这里 |
| `retrieval/advance_retriever.py`             | 89 KB  | `MultiRetriever`（L52）与 `smart_search` 主链路     |
| `retrieval/rerank_manager.py`                | 85 KB  | 重排器管理（懒加载单例），性能敏感                  |
| `core/semantic_graph.py`                     | 84 KB  | 图结构与门面 API（"关系约束"方向的挂载点）          |
| `retrieval/bm25_retriever.py`                | 84 KB  | 自研倒排 + numba；`candidate_uids` 过滤范例         |
| `retrieval/splade_retriever.py`              | 73 KB  | 稀疏向量检索与候选行过滤                            |
| `triple_retrieval/triple_tower_retriever.py` | 55 KB  | 三塔编排总入口（L202）                              |
| `auto_builder/orchestrator.py`               | 47 KB  | 高阶记忆构建编排                                    |
| `retrieval/graph_context_expander.py`        | 46 KB  | 检索后图扩展                                        |
| `quantification/cascade_pruner.py`           | 43 KB  | 级联剪枝（量化检索核心）                            |

**步骤 4：项目形态与入口。** Mandol 是**库（主）+ 示例应用 + 评测脚本**三位一体：


| 入口            | 位置                                                         | 读法                                                                                                          |
| ----------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 库导出面        | `src/mandol/__init__.py`（L34-53）                           | 只导出 4 个核心类；`SemanticMap/SemanticGraph` 懒加载（避免 import 即加载 Torch/FAISS）                       |
| README 快速示例 | `README_CN.md:275-310`                                       | 最短主干：`SemanticMap` → `SemanticGraph` → `add_unit` → `search_similarity_in_graph` → `save/load_graph` |
| Web 示例        | `examples/mandol_chat/run.py`                                | `uvicorn mandol_chat.main:app`；`main.py` 是 FastAPI 装配点，`services/mandol_service.py` 是 Mandol 集成点    |
| 评测脚本        | `benchmark_locomo/task_eval/locomo_benchmark_episodic.py` 等 | task_eval 直连`MultiRetriever.smart_search`（L157）——"官方怎么用"的活样例                                   |

### 1.2 模块地图（src/mandol 13 个子包）


| 子包                                           | 职责一句话             | 关键类                                                                                                                                                               |
| ------------------------------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core/`                                        | 记忆模型与融合存储     | `MemoryUnit`、`SemanticMap`、`SemanticGraph`、`MemorySpace`                                                                                                          |
| `retrieval/`                                   | 进程内多路检索管线     | `BaseRetriever` 族、`MultiRetriever`（advance_retriever.py:52）、`CosineRetrieverAdapter`（cosine_retriever.py:19）、`ScoreFusion`、`RerankerManager`、`QueryBundle` |
| `triple_retrieval/`                            | 三塔编排与量化打包     | `TripleTowerRetriever`（L202）                                                                                                                                       |
| `hierarchical/` `episodic/` `entity_relation/` | 三塔各自的检索器       | `HierarchicalRetriever` 等                                                                                                                                           |
| `memory_router/`                               | 查询意图路由           | `LocomoTowerRouter`、`LongMemEvalTowerRouter`                                                                                                                        |
| `quantification/`                              | 充分性判断与上下文打包 | `SemanticQuantifier`、`CascadePruner`                                                                                                                                |
| `auto_builder/`                                | 高阶记忆构建           | `Orchestrator`、`HierarchicalBuilder`、`EpisodicBuilder`、`EntityRelationBuilder`                                                                                    |
| `storage/`                                     | RocksDB 分层持久化     | `TieredStorageManager`、`RocksDBPayloadStore`                                                                                                                        |
| `llm/`                                         | LLM 客户端             | `LLMClient`、`BaseProvider`（模板方法）                                                                                                                              |
| `cluster/`                                     | 图/节点聚类            | `BaseClusterer`、`create_clusterer()`                                                                                                                                |
| `utils/`                                       | 日志等基础设施         | `create_module_logger`                                                                                                                                               |

### 1.3 关键配置文件解读（本仓库专属结论）


| 配置                      | 读出的关键事实                                                                                                                                                                                                                                                                                               |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`          | Python**3.12 only**（`>=3.12,<3.13`，L31-32 运行时还会强制校验）；默认依赖即"论文完整栈"：`faiss-cpu`、`rustworkx`、`rocksdict`、`numba`、`torch==2.8.0`（锁定）、`transformers 4.x`；extras：`cuda`（flash-attn）、`dev`（pytest/ruff）、`docs`（sphinx + mermaid）；pytest marker `real_model`（L161-164） |
| `uv.lock`                 | 真实版本以锁定文件为准（pyproject 中的是范围）；项目命令统一走`uv run`（Makefile）                                                                                                                                                                                                                           |
| `env.template`            | 三类开关：① LLM/HF 凭据（DEEPSEEK/OPENAI/…、HF_TOKEN）；② 重排后端`RERANKER_BACKEND=native|vllm`；③ 调优项 `MANDOL_BM25_SPACY_MAX_PROCESSES`、`SKIP_AUTO_LOGGING`。**核心库不配 env 也可跑**（本地 embedding），重排/LLM 功能受限                                                                        |
| `Makefile`                | 测试入口：`make test` = `tests/test_rocksdb_tiered_cache.py` + `examples/mandol_chat/tests`；`make lint`（ruff）、`make syntax`、`make docs`、`make build`。**注意：全仓测试主战场在 examples 里，根 `tests/` 只有 1 个文件**                                                                                |
| `.pre-commit-config.yaml` | 提交钩子（对应 CONTRIBUTING 的 Ruff、行长 100 约定）                                                                                                                                                                                                                                                         |

导引三要点在本仓库的落点：**默认值覆盖顺序**（未 `connect_to_l2` 时 payload 全驻留内存）、
**能力开关**（extras、`RERANKER_BACKEND`、`generate_sparse_embedding=False`）、**lock 版本为准**（uv.lock）。

### 1.4 依赖与技术栈识别（贴标签）


| 角色       | 依赖                                                        | 在 Mandol 中的角色                                                 |
| ------------ | ------------------------------------------------------------- | -------------------------------------------------------------------- |
| 语言运行时 | Python 3.12                                                 | 用了`__slots__`、`@dataclass`、延迟导入（`__getattr__`）等         |
| 向量索引   | `faiss-cpu`                                                 | `SemanticMap` 稠密索引（IndexIDMap + 内部 int-id）                 |
| 图引擎     | `rustworkx`（+ networkx / igraph / leidenalg）              | `SemanticGraph` 主结构；igraph/leidenalg 供 cluster                |
| 持久化     | `rocksdict`（RocksDB 绑定）                                 | storage 分层换页（唯一正式 payload backend）                       |
| 加速       | `numba`                                                     | BM25 倒排计算加速                                                  |
| ML         | torch 2.8 + transformers 4.x + sentence-transformers        | 本地 embedding / rerank（`use_flash_attention=False` 即 CPU 路径） |
| Web        | fastapi + uvicorn                                           | `examples/mandol_chat`                                             |
| LLM 客户端 | openai / tenacity / tiktoken / dashscope                    | `llm/` 与 quantification 的 LLM 调用（tenacity = 重试模板）        |
| NLP        | jieba / rank-bm25 / spaCy / nltk / langchain-text-splitters | BM25 分词与文本处理                                                |
| 评测       | bert-score / rouge-score / tabulate / pandas                | benchmark_*                                                        |

### 1.5 粗读记录清单（已填写——阅读者请逐项核对、补充自己的疑问）

- [X]  项目用途一句话：为 LLM 智能体提供"内存为中心"的分层记忆存储与高精度检索（README_CN:52）
- [X]  项目形态：库（主）+ Web 示例 + 评测脚本
- [X]  架构模式：融合存储（键值 + 向量 + 图）+ 多路召回/分数融合检索管线 + 可选高阶编排（路由/三塔/量化）
- [X]  核心功能模块 Top 5：① `core` 融合存储 ② `retrieval` 多路检索与融合 ③ `storage` 分层换页 ④ `triple_retrieval` 三塔编排 ⑤ `quantification` 量化
- [X]  技术栈（见 1.4）
- [X]  入口清单（见 1.1 步骤 4）
- [X]  主干调用链（粗粒度）：`add_unit` → FAISS/BM25/SPLADE 增量维护 →（检索时）`smart_search` → 三路并行 → RRF → 重排 →（可选图扩展）→ `[(MemoryUnit, score)]`
- [X]  配置要点：Python 3.12 + uv；env 可选；Makefile 为命令入口
- [X]  遗留疑问（带入精读，均有答案）：① `candidate_uids` 如何到达三路后端？② 换页后索引为何仍可用？③ 三塔与 `smart_search` 什么关系？④ 图 API 为什么会 `AttributeError`？

---

## 二、精读阶段

### 2.1 核心模块识别与精读优先级


| 优先级 | 模块                                             | 识别信号                                 |
| -------- | -------------------------------------------------- | ------------------------------------------ |
| P0     | `core/semantic_map.py`                           | 体量第一、被全仓依赖、承载所有数据不变量 |
| P0     | `retrieval/advance_retriever.py`                 | 检索主链路（`smart_search`），性能核心   |
| P0     | `core/semantic_graph.py`                         | 图门面 API 与关系维护                    |
| P1     | 四个检索后端（bm25/splade/cosine/graph）         | 候选过滤与评分实现                       |
| P1     | `storage/tiered_storage_manager.py`              | 与 core 换出钩子成对阅读                 |
| P2     | triple_retrieval / quantification / auto_builder | 高阶编排，按方向任务再深入               |

反向排除（快速略读）：`website/`、`docs/archive/`、`picture/`、`release-notes/`、`cuda_stream_utils` 等纯胶水工具。

### 2.2 关键算法与数据结构（五问速查表）


| 算法 / 结构           | 输入→输出           | 复杂度要点                           | 结构选择理由                                     | 不变量                        | 边界                                                                         |
| ----------------------- | ---------------------- | -------------------------------------- | -------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------ |
| FAISS 稠密检索        | query 向量→TopK uid | 全量或子集检索；子集依赖 int-id 集合 | IndexIDMap 绑定稳定 int-id，支持 remove/add 增量 | `uid ↔ int_id` 双向映射一致  | 零向量跳过；`remove_ids` 失败→**全量 rebuild**（semantic_map.py:1075-1121） |
| BM25 自研倒排         | tokens→评分         | numba 加速内层循环                   | 倒排数组 + 内部 id                               | 倒排 id 与 uid 映射一致       | `candidate_uids` 转内部 id 过滤（bm25_retriever.py:1499-1574）               |
| SPLADE 稀疏检索       | 稀疏向量→评分       | 静态矩阵行打分                       | 静态矩阵 + 行索引                                | 行 id 稳定                    | 候选行过滤（splade_retriever.py:959-997）                                    |
| RRF / 加权 / MMR 融合 | 多路列表→单列表     | O(Σn log n)                         | RRF 无参稳健；MMR 去冗余                         | 分数单调性                    | 单路缺失时降级                                                               |
| rustworkx 图          | 节点/边查询→子图    | BFS O(V+E)                           | 原生图结构；边属性 LRU 热缓存                    | `_uid_to_index` 双向一致      | **三处断链 API 勿用**（§4.4-1）                                             |
| 分层换页              | 容量水位→驻留集合   | 冷选排序 O(n log n)                  | 访问计数 + 时间戳排序                            | 索引/映射/空间/图**常驻不换** | 只删 payload（semantic_map.py:631-653）                                      |
| 空间过滤缓存          | 空间名→uid 集合     | 缓存 + 版本号失效                    | `_space_membership_version`                      | 版本变更即失效                | `_get_candidate_uids_set`；BM25 侧 `_get_space_filter_internal_ids`          |

### 2.3 复杂实现阅读技巧（本仓库实战建议）

- **semantic_map.py 不要顺序通读**（136 KB）。按"三条流程"跳读：
  写入（`add_unit` L1280、`batch_add_units` L1376、`_apply_index_update_mode` L1264-1278）、
  过滤（`filter_memory_units` L3169）、换页（`_trigger_tiered_eviction_if_needed` L605、
  `_remove_from_l1_for_tiered_swap` L631）。用 IDE outline 按方法名跳转。
- **advance_retriever.py 顺 kwargs 读**：先 `smart_search` 顶层 → `_execute_base_retrieval`
  （L1154-1195：`space_names` 显式、其余走 kwargs）→ 注意入口剔除 `GRAPH_TRAVERSAL`（L937）、
  `_ensure_retriever_loaded` 的 if/elif 加载链（L150-190）。
- **"追踪一个真实输入"**：拿 README 示例的 `msg_001` 手工走一遍
  `add_unit("张三今天去了北京。")`：`raw_data` → `text_cached` 派生（memory_unit.py:56-65）→
  embedding → FAISS int-id → BM25 tokens →（`generate_sparse_embedding=False` 则跳过 SPLADE）。
- **去噪**：`**legacy_kwargs` 兼容参数、logger 行、参数校验块第一遍全部跳过，读完主干再回读。

### 2.4 重要接口与数据流

**接口清单（精读必记）：**


| 接口                                                   | 位置                                                                                                                                                                         | 契约要点                                                                                     |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `SemanticMap.add_unit / batch_add_units / delete_unit` | 1280 / 1376 / 1817                                                                                                                                                           | 写入即增量维护三索引；同 uid 同内容跳过；`index_update_mode` ∈ {incremental, none, rebuild} |
| `SemanticMap.filter_memory_units`                      | 3169                                                                                                                                                                         | 10 操作符；取值`getattr → raw_data`；**不查 metadata**；无分页                              |
| `SemanticGraph` 门面族                                 | `add_unit`:376 / `search_similarity_in_graph`:1049 / `search_graph_relations`:1134 / `get_node_neighbors`:1183 / `connect_to_l2`:182 / `save_graph`:1626 / `load_graph`:1852 | 与 README 示例一一对应；注意 1134/1183 内部断链                                              |
| `MultiRetriever.smart_search`                          | advance_retriever.py                                                                                                                                                         | `**kwargs` 透传 `candidate_uids` / `space_names`；融合 + 重排 + 图扩展                       |
| `BaseRetriever.search`                                 | retrieval_interface.py                                                                                                                                                       | 子类显式签名`search(query, top_k, candidate_uids=None, space_names=None)`                    |
| `TripleTowerRetriever.search`                          | triple_tower_retriever.py:202                                                                                                                                                | 三塔分发、统一重排、量化打包                                                                 |

**数据流（三条主干）：**

1. **写入流**：`MemoryUnit(raw_data)` → dense embedding → `memory_units[uid]` → FAISS/BM25/SPLADE 增量
   （`_apply_index_update_mode`）→ 换页检查；
2. **检索流**：query → `QueryBundle` 编码缓存 → 三路并行（`candidate_uids` 过滤）→ `ScoreFusion` →
   Reranker → GraphExpander → `[(MemoryUnit, score)]`；
3. **换页流**：水位触发 → 冷 uid 排序 → 删 payload（保留索引）→ 检索命中时 page-in。

**跨边界重点**（问题高发区）：`save_graph/load_graph` 序列化点、RocksDB 读写点、
FAISS↔uid 映射、JSON↔张量化。

### 2.5 异常与边界（本仓库已核对清单）

- **FAISS**：`remove_ids` 不支持/异常 → fallback **全量 rebuild**（1075-1121）；零向量跳过；
  embedding 生成失败仍入库但不可向量检索（L1345-1350）；
- **参数兼容**：`rebuild_index_immediately` 等旧参数走 `**legacy_kwargs` 兼容，未知参数直接
  `TypeError`（L1313-1318）；
- **图**：三处断链调用运行必抛 `AttributeError`（§4.4-1）；
- **换页**：未调用 `connect_to_l2()` 时 payload 全驻留；启用后 RocksDB 写入与 resident 删除
  **可能异步**（README_CN:326）；
- **并发**：`QueryBundle` 加锁缓存查询编码；换页与检索并发时以 `_storage_uids` 判断 payload 归属；
- **版本**：Python < 3.12 直接 `RuntimeError`（`__init__.py:31-32`）。

---

## 三、辅助理解工具与方法（本仓库适用版）

### 3.1 注释与规范解读

- docstring 为英文、Google 风格变体；模块头统一说明职责；日志统一
  `create_module_logger("模块名")`（utils/logging_config）；
- 本仓库注释与代码一致度较高（如 FAISS fallback 的说明），可作为可靠线索——但 §4.4-1 提醒：
  **调用点存在 ≠ 被调方法存在**，注释/签名之外必须核验被调方；
- 标记注释用 `rg "TODO|FIXME|XXX|HACK"` 单独收集成线索清单。

### 3.2 调试与运行工具（本仓库命令）

- 环境：`uv sync --extra dev --group spacy-model`（= `make dev`）；
- 测试：`make test` / `make test-unit` / `make test-integration`；
- **离线单测范式**：`tests/test_rocksdb_tiered_cache.py:24-81`（monkeypatch `global_model_manager`
  + Dummy 模型注入）——不联网也能测，新增测试照此办理；
- 规范：`make lint`（ruff）、`make syntax`（compileall）；
- 日志与调优：嵌入宿主应用设 `SKIP_AUTO_LOGGING=true`；BM25 并行分词进程数
  `MANDOL_BM25_SPACY_MAX_PROCESSES`；
- **最小复现**：优先用 README 示例（`all-MiniLM-L6-v2` + `use_flash_attention=False`，CPU 可跑）
  + `generate_sparse_embedding=False` 跳过 SPLADE 模型下载。

### 3.3 文档与注释交叉验证（本仓库已做，结论如下）


| 文档"说"                                                                      | 代码"做"                                                                   | 判定                      |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------- |
| README 示例 API（`add_unit` / `search_similarity_in_graph` / `save_graph`…） | semantic_graph.py:376 / 1049 / 1626 均存在、签名匹配                       | 一致 ✓                   |
| README "RocksDB 是唯一正式支持的 persistent backend"                          | storage/ 仅有`rocksdb_payload_store` + `tiered_storage_manager`            | 一致 ✓                   |
| README "换出后…payload materialization 发生在 search 调用内"                 | `_remove_from_l1_for_tiered_swap` 只删 payload；page-in 在 `get_unit` 路径 | 一致 ✓（异步细节待验证） |
| `search_graph_relations(seed_nodes=…)` / `get_node_neighbors` 看似可用       | 内部依赖`edge_bfs_search` / `get_relevant_nodes` **未定义**                | **代码缺陷**（§4.4-1）   |
| 文档/导引预期根`tests/` 是测试主战场                                          | 实际只有 1 个文件；主力测试在`examples/mandol_chat/tests`（Makefile 指定） | 以 Makefile 为准          |

方法示范：任何"看起来能用"的 API，至少做一次**调用点 → 被调方存在性**核验（本仓库最大教训即此）。

### 3.4 流程图建议（本仓库推荐图型）


| 想表达     | 推荐图   | 对应内容                                                                  |
| ------------ | ---------- | --------------------------------------------------------------------------- |
| 模块依赖   | 框图     | §4.1 文字版，可转 mermaid（docs 含 sphinxcontrib-mermaid）               |
| 检索时序   | 时序图   | query →`MultiRetriever` → 三路 → `ScoreFusion` → Reranker → Expander |
| 写入控制流 | 流程图   | `add_unit` + `index_update_mode` 三分支                                   |
| 换页状态   | 状态机图 | resident ↔ persistent，watermark 触发                                    |

---

## 四、阅读理解成果输出

### 4.1 架构图（三层）

**（1）系统上下文图（文字版）：**

```
                   ┌─────────────── 应用 / Agent ───────────────┐
                   │  README 示例   |   examples/mandol_chat    │
                   └──────────────────┬─────────────────────────┘
                公开 API：MemoryUnit / SemanticMap / SemanticGraph
                                    │
        ┌───────────────────────────┼─────────────────────────────┐
        ▼                           ▼                             ▼
  core 融合存储             retrieval 检索管线             storage 分层持久化
  (Map+Graph+Space)      (BM25/SPLADE/Cosine+融合+重排)    (RocksDB tiered cache)
        ▲                           ▲
        └─ 高阶编排（可选）：triple_retrieval / memory_router / quantification / auto_builder / llm
```

**（2）模块/组件图**：用 §1.2 表格 + 依赖方向（`core` ← `retrieval` ← `triple_retrieval`；
`storage` 经 core 换出钩子接入）。

**（3）关键流程时序图**：写入 / 检索 / 换页三条，素材见 §2.4 数据流。

### 4.2 核心模块说明卡（已填 5 张，可直接扩展）

**卡片 1：core / SemanticMap**

```
一句话职责：为全系统提供"内存为中心"的记忆融合存储——键值 + 稠密/稀疏向量索引 + 空间成员，
          并管理冷热 payload 换页。
关键文件/类：core/semantic_map.py 的 SemanticMap
对外接口：add_unit(1280) / batch_add_units(1376) / delete_unit(1817) / get_unit /
         filter_memory_units(3169) / build_index / rebuild_all_indexes
内部关键流程：嵌入生成 → payload 写入 → 三索引增量（_apply_index_update_mode）→ 换页检查；
           换出/换入钩子（605-670）
依赖：FAISS；BM25/SPLADE（经 _multi_retriever）；TieredStorageManager（可选）
数据存储/状态：memory_units、_uid_to_int_id、_space_membership_version、_access_counts、_storage_uids
验证方式：tests/test_rocksdb_tiered_cache.py（换页）+ README 示例（写入/检索）
已知问题：职责过重（上帝类）；filter 线性扫描且不查 metadata
```

**卡片 2：core / SemanticGraph**

```
一句话职责：以 rustworkx 图承载记忆间关系，并向上层提供门面 API。
关键文件/类：core/semantic_graph.py 的 SemanticGraph
对外接口：add_relationship(610) / add_unit(376) / search_similarity_in_graph(1049) /
         connect_to_l2(182) / save_graph(1626) / load_graph(1852)
内部关键流程：关系双向建边；边属性 LRU 热缓存；门面转发到 SemanticMap
已知问题：1134 / 1126-1132 / 1183 三处断链 API（§4.4-1）
```

**卡片 3：retrieval / MultiRetriever**

```
一句话职责：多路召回的调度与融合（RRF/加权/MMR + 重排 + 图扩展）。
关键文件/类：retrieval/advance_retriever.py:52
对外接口：smart_search(query, top_k, **kwargs)
内部关键流程：剔除 GRAPH_TRAVERSAL(937) → QueryBundle 编码缓存 → _execute_base_retrieval(1154-1195)
           → 三路并行 → ScoreFusion → Reranker → Expander
已知问题：_ensure_retriever_loaded(150-190) 为 if/elif 加载链（扩展需改枚举+链，OCP 打折）
```

**卡片 4：storage / TieredStorageManager**

```
一句话职责：基于水位线的冷 payload 换页调度（RocksDB 持久层）。
关键文件/类：storage/tiered_storage_manager.py、rocksdb_payload_store.py
对外接口：check_and_trigger_eviction（core 在 add 路径调用）；core 侧回调 swap_out / swap_in
数据存储/状态：max_capacity / high_watermark / low_watermark；_storage_uids 目录（core 侧）
验证方式：tests/test_rocksdb_tiered_cache.py
```

**卡片 5：triple_retrieval / TripleTowerRetriever**

```
一句话职责：三塔（层级/图/情景）编排 + 统一重排 + 量化打包。
关键文件/类：triple_retrieval/triple_tower_retriever.py:202、TripleTowerConfig / 各 TowerResult
依赖：三塔检索器 + memory_router + RerankerManager + CascadePruner
备注：quantification（token 受限上下文）不在 smart_search 管线内，挂在 _build_quantification
```

### 4.3 关键代码片段分析示例（本仓库真实代码）

对 `_remove_from_l1_for_tiered_swap`（semantic_map.py:631-653）按"契约 → 主干 → 细节"分析：

```python
def _remove_from_l1_for_tiered_swap(self, uids: List[str]) -> int:
    """Remove payloads while preserving all resident retrieval and graph state."""
    if not uids:
        return 0

    removed_count = 0
    for uid in uids:
        if uid in self.memory_units:
            del self.memory_units[uid]
            removed_count += 1
        self._modified_units.discard(uid)
        self._access_counts.pop(uid, None)
        self._last_accessed.pop(uid, None)
        if self._storage_uids is not None:
            self._storage_uids.add(uid)

    if removed_count:
        logger.info("Tiered storage evicted %d payloads; indexes, UID mappings, "
                    "MemorySpace membership, and graph topology remain resident.",
                    removed_count)
    return removed_count
```


| 观察层                | 结论 / 疑问                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 契约（签名层）        | 入参 uid 列表，返回实际删除数；副作用：删 payload、清访问记账、把 uid 写入`_storage_uids` 目录；docstring 明示"保留全部检索与图状态"                 |
| 主干（控制流）        | 循环内四件事：`del` payload / `discard` 脏标记 / 清访问计数 / 登记持久层目录                                                                         |
| 与索引的关系          | FAISS int-id 映射、空间成员、图拓扑**均未触碰**——检索仍会命中该 uid，命中后由 page-in（655-670）恢复 payload。这是"payload 级换页"设计的核心证据   |
| 边界                  | 空列表直接返回 0；uid 不存在也安全（`pop/discard` 容错）；`_storage_uids` 为 None（未启用 L2）时不登记                                               |
| 待验证（并发/一致性） | `_storage_uids.add` 与真正的 RocksDB 写入是否同事务？README_CN:326 说换出"可能异步"——崩溃窗口内目录与持久层是否一致，值得写最小用例验证（§4.4-5） |
| 数据流                | `memory_units`（resident dict）→ RocksDB（persistent）；`_storage_uids` 是目录索引，不是数据本体                                                    |

**练习建议**：用同样的三层读法分析 `_execute_base_retrieval`（advance_retriever.py:1154-1195），
重点标注 `space_names` 显式参数与 `candidate_uids` kwargs 的路径差异。

### 4.4 潜在问题记录（本仓库初步清单）


| # | 位置                                      | 现象                                                                              | 触发条件                                                            | 影响                                       | 证据                               | 置信度           | 建议动作                                               |
| --- | ------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------ | ------------------ | -------------------------------------------------------- |
| 1 | semantic_graph.py:1161 / 1126-1132 / 1204 | 调用未定义方法（`edge_bfs_search` / `hybrid_node_search` / `get_relevant_nodes`） | 调用`search_graph_relations(seed_nodes=…)` 或 `get_node_neighbors` | 运行时`AttributeError`，相关关系检索不可用 | 全仓库仅调用点、无定义（静态核验） | 确认（静态）     | 写最小用例复现；修复或绕过（见《优化方向.md》2.5）     |
| 2 | semantic_map.py:3169                      | `filter_memory_units` 线性扫描、不查 `metadata`、无分页                           | 大库 + 属性过滤                                                     | 过滤 O(N)；metadata 字段查不到             | 源码逐行                           | 确认             | 组合查询需另建索引（见方案）                           |
| 3 | semantic_map.py:3054                      | `created_time` 写入路径不填充（默认 None）                                        | 任何`add_unit`                                                      | 字段语义不可用，易误用                     | 源码                               | 确认             | 时间约束改用`raw_data["timestamp"]`                    |
| 4 | advance_retriever.py:1154-1195            | 扩容`max(top_k*3, 50)` 为硬编码                                                   | 高选择性约束 + post-filter                                          | 大量结果被过滤后可能凑不满 top_k           | 源码                               | 高度疑似         | 实验：候选占比曲线下测召回缺口                         |
| 5 | semantic_map.py:631-653 + README_CN:326   | 换出目录登记与异步 RocksDB 写入存在时间窗                                         | 换页过程中崩溃                                                      | 目录与持久层可能不一致                     | 源码 + README                      | 存疑             | 阅读 TieredStorageManager 确认写入时机，写故障注入用例 |
| 6 | advance_retriever.py:150-190              | 检索器加载链为 if/elif                                                            | 新增召回源                                                          | 需同时改枚举与加载链                       | 源码                               | 确认（设计观察） | 若确需扩展，评估注册表模式                             |
| 7 | core/semantic_map.py 全文件               | `SemanticMap` 职责过重（存储+索引+过滤+序列化，136 KB）                           | 任何改动                                                            | 维护与测试成本高                           | 体量 + 方法分布                    | 确认（设计观察） | 新能力以组合方式接入，勿继续加责                       |

**纪律**（导引 §4.4）：只记可证实项；推断标置信度；每条配验证动作；定期把"存疑"收敛为"确认/排除"。

### 4.5 阅读检查点（读完应能口头回答）

1. 一条 `MemoryUnit` 从 `add_unit` 到可检索，经过了哪几个索引的哪些维护？
2. `candidate_uids` 如何从 `smart_search` 一路透传到 FAISS / BM25 / SPLADE？
3. 换页删掉了什么、保留了什么？为什么检索仍能命中冷数据？
4. `smart_search` 为什么不包含 `GRAPH_TRAVERSAL`？图证据如何进入结果？
5. 三塔（`triple_retrieval`）与 `smart_search` 的关系？`quantification` 挂在哪一层？
6. 关系维护的唯一入口是哪个方法？
7. `filter_memory_units` 的字段取值顺序？为什么不查 `metadata` 是个问题？
8. 三处断链 API 分别在哪、影响了哪些公开方法？
9. 时间戳实际存在哪里？为什么不能用 `created_time`？
10. 若要新增"组合约束"能力，应该改哪一层、不需要改哪一层？（参见《优化方向.md》）

---

## 附录 A：命令速查（本仓库）


| 目的            | 命令                                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| 同步开发环境    | `uv sync --extra dev --group spacy-model`                                                          |
| 跑测试          | `make test`（或 `uv run pytest tests/test_rocksdb_tiered_cache.py examples/mandol_chat/tests -v`） |
| Lint / 语法检查 | `make lint` / `make syntax`                                                                        |
| 构建文档        | `make docs`                                                                                        |
| 找检索主链路    | `rg "def smart_search                                                                              |
| 找候选集下推    | `rg "candidate_uids" src/mandol`                                                                   |
| 找待办与隐患    | `rg "TODO                                                                                          |
| 找变更热点      | `git log --name-only --pretty=format: | sort | uniq -c | sort -rn | head -20`                      |

## 附录 B：术语速查


| 术语                   | 含义                                                                     |
| ------------------------ | -------------------------------------------------------------------------- |
| L1 / L2                | L1 = 内存驻留 payload；L2 = RocksDB 持久层（`connect_to_l2()` 启用换页） |
| payload                | `MemoryUnit` 本体数据（与"索引"相对）                                    |
| page-in / swap-out     | 冷 payload 落盘 / 命中后唤回内存                                         |
| 三塔                   | hierarchical（层级）/ graph（实体关系）/ episodic（情景）三条记忆检索塔  |
| 量化（quantification） | token 约束下的充分性判断、剪枝与上下文打包                               |
| RRF / MMR              | 倒数排名融合 / 最大边际相关性（去冗余）                                  |
| 门面（Facade）         | `SemanticGraph` 对 `SemanticMap` 能力的包装转发                          |
| 断链 API               | 有调用点、无定义的三个图方法（§4.4-1）                                  |
