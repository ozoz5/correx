# CORREX

Claude や他の LLM フロントエンドから使うための MCP サーバーです。
あなたの修正が蓄積され、ルールになり、次の生成に注入されます。

> **Engram エンジン**搭載 — 実際のやりとりから行動パターンを抽出し、Rules → Meanings → Principles に昇格、次のセッション前に注入します。

## 入っているもの

- `HistoryStore` — 会話・修正・エピソードの永続化
- `Engram engine` — ルール昇格・矛盾解消・自己超克
- `PersonalityLayer` — 行動プロファイル（metabolism_rate, reward_pattern など）
- `MCP server` — Claude Code / Claude Desktop から直接使える
- `Training dataset / auto-train` — accepted な出力を教師データ化し、`mlx-lm` で LoRA 学習

## 使い方

```bash
git clone https://github.com/ozoz5/correx
cd correx
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[mcp]"
```

## Claude Code への追加

```bash
claude mcp add correx \
  -s user \
  -- /path/to/correx/.venv/bin/python \
  -m correx \
  --memory-dir ~/.correx \
  --transport stdio
```

## Claude Desktop への追加

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "correx": {
      "command": "/path/to/correx/.venv/bin/python",
      "args": [
        "-m", "correx",
        "--memory-dir", "/Users/you/.correx",
        "--transport", "stdio"
      ]
    }
  }
}
```

## 主な MCP ツール

- `build_guidance_context` — 蓄積ルールをコンテキストに注入
- `save_conversation_turn` — 修正・承認を記録
- `rebuild_preference_rules` — パターンをルールに昇格
- `synthesize_meanings` — ルール群から深いパターンを抽出
- `synthesize_principles` — Meanings から原則を蒸留
- `get_personality_profile` — 行動プロファイルと自己超克提案
- `synthesize_rules` — 成功/失敗パターンからルール仮説を生成
- `record_growth` — 品質改善の前後を記録
- `save_correction` — 手直しを記録
- `save_episode` — タスク結果を保存

## フォルダ構成

```
correx/
  src/correx/
    mcp_server.py          # MCP ツール定義
    service.py             # コアサービス層
    history_store.py       # 永続化 + I/O オーケストレーション
    rule_builder.py        # 純粋なルール構築ロジック（I/O なし）
    memory_manager.py      # 矛盾解消・自己補正
    meaning_synthesis.py   # Engram: Rules → Meanings → Principles
    personality_layer.py   # 行動プロファイリング
    llm_scorer.py          # 反応スコアリング（Anthropic API / ルールベース）
  tests/                   # 90 テスト通過
```

全データは `~/.correx/` に JSON で保存。データベース不要。

## LoRA 学習（オプション）

```bash
pip install -e ".[train]"

python3 scripts/auto_train.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --memory-dir ~/.correx \
  --output-dir ./training_artifacts
```

## 意図

これは Claude 専用の prompt 集ではありません。
**Claude が使う知性 OS の核**です。

AGI が万人のための汎用知性なら、CORREX は一人のための超越知性。
