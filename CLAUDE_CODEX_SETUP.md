# Claude / Codex MCP Setup

## 概要

`claude_pseudo_intelligence_core` は、Claude と Codex の両方から同じ MCP サーバとして利用できる。
この構成では、両クライアントが同じ memory directory を読むので、

- correction
- promoted rule
- guidance context
- training example

を共有できる。

中核はこのサーバ:

- `/Users/setohirokazu/Desktop/claude_pseudo_intelligence_core/scripts/run_mcp_server.py`

共有メモリはここ:

- `/Users/setohirokazu/.pseudo-intelligence`

## 現在の状態

### Claude

Claude 側はすでに接続済み。

- 設定ファイル: `/Users/setohirokazu/Desktop/疑似知性/.claude/settings.json`
- MCP 名: `pseudo-intelligence`

### Codex

Codex 側も接続済み。

- 設定ファイル: `/Users/setohirokazu/.codex/config.toml`
- MCP 名: `pseudo-intelligence`

## Codex 設定

`~/.codex/config.toml` に入っている設定:

```toml
[mcp_servers."pseudo-intelligence"]
command = "/opt/homebrew/bin/python3"
args = [
  "/Users/setohirokazu/Desktop/claude_pseudo_intelligence_core/scripts/run_mcp_server.py",
  "--memory-dir",
  "/Users/setohirokazu/.pseudo-intelligence",
  "--transport",
  "stdio",
]
```

サンプルはここにもある:

- `/Users/setohirokazu/Desktop/claude_pseudo_intelligence_core/examples/codex_config.toml`

## Claude 設定

Claude 側は `settings.json` 内で次を使っている:

- MCP server: `pseudo-intelligence`
- memory dir: `/Users/setohirokazu/.pseudo-intelligence`
- transport: `stdio`

つまり、Claude と Codex は **同じサーバ実装** と **同じ保存先** を共有する。

## 起動後にやること

設定変更後は、各クライアントを再起動する。

### Claude

- Claude を開き直す

### Codex

- Codex アプリを開き直す

## 動作確認

再起動後、MCP ツールとして次が見えれば接続成功:

- `build_guidance_context`
- `save_episode`
- `save_correction`
- `save_conversation_turn`
- `list_preference_rules`

また、resource として次が読めればよい:

- `memory://summary`
- `memory://guidance/{task_scope}`

## いま出来ること

この状態で、

1. Claude で correction を保存する
2. Codex で guidance を引く
3. Codex で新しい correction を保存する
4. Claude でその rule を使う

という往復ができる。

つまり、クライアントは別でも、学習資産は一つにできる。

## まだ足りないもの

接続自体は完了しているが、`会話するたびに自動成長` はまだ薄い。

次に必要なのはこれ:

1. 対話前に `build_guidance_context` を自動で呼ぶ
2. 修正後に `save_conversation_turn` を自動で呼ぶ
3. 完了時に `save_episode` を自動で呼ぶ

つまり、

**MCP 接続は完了。  
自動学習ループは次の段。**

## 一言で言うと

いまは、

**Claude にも Codex にも同じ AI Correction OS を刺せる状態**

まで来ている。
