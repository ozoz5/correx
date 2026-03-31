# Claude Pseudo Intelligence Core 説明書

## これは何か

`claude_pseudo_intelligence_core` は、Claude や他の LLM に

- 過去の仕事を記録させる
- 人間の手直しを学習信号として残す
- 次の仕事で似た補正だけを引く

ための最小コアです。

一言で言うと、

**「その場で答えるAI」を「直しから少しずつ強くなるAI」へ変える外部記憶層**

です。

Claude 自体のモデルを学習させるのではなく、Claude の外に

- `episode memory`
- `correction memory`
- `conversation memory`
- `preference rules`
- `guidance context`

を置き、毎回それを使わせる設計です。

## なぜ別フォルダにしたか

元の `疑似知性` プロジェクトは、ゲーム、bench、worktree 実験、UI などが広く入っています。  
Claude に汎用スキルとして載せるには重すぎる。

そこでこのフォルダでは、擬似知性のうち本当に汎用な部分だけを抜いています。

- 記録
- 手直し
- 会話修正
- 再利用
- 秘密保管

これだけです。

## 何が入っているか

### 1. `HistoryStore`

ファイル: [history_store.py](/Users/setohirokazu/Desktop/claude_pseudo_intelligence_core/src/claude_pseudo_intelligence/history_store.py)

役割:

- タスク結果を `EpisodeRecord` として保存する
- 人間の手直しを `CorrectionRecord` として保存する
- 似たタスクをあとから引けるようにする

保存される主な情報:

- `title`
- `issuer`
- `task_type`
- `source_text`
- `company_profile`
- `output`
- `corrections`

### 2. `Learning Context`

ファイル: [learning_context.py](/Users/setohirokazu/Desktop/claude_pseudo_intelligence_core/src/claude_pseudo_intelligence/learning_context.py)

役割:

- 今回のタスクと近い過去案件を探す
- その案件で人間がどう直したかを抽出する
- 会話で繰り返し出た好みや禁止事項を抽出する
- Claude の prompt に差し込める形へ整形する

出てくるものは、例えばこういう文脈です。

```text
# HUMAN CORRECTION MEMORY
## Previous case: 自治体DX提案のGo/No-Go判断 / 東京都
- Why relevant: same_domain | same_issuer
- Actual override: 条件付き参加
- Human correction: 価格競争より移行体制を前面に出すべきだった
- Reuse guidance: 運用定着と引継ぎ体制を先に見せる
```

### 3. `Secret Store`

ファイル: [secret_store.py](/Users/setohirokazu/Desktop/claude_pseudo_intelligence_core/src/claude_pseudo_intelligence/secret_store.py)

役割:

- macOS なら Keychain に秘密を保存する
- API キーや接続トークンを平文ファイルに置かない

### 4. `Conversation Memory / Preference Rules`

ファイル: [conversation_learning.py](/Users/setohirokazu/Desktop/claude_pseudo_intelligence_core/src/claude_pseudo_intelligence/conversation_learning.py)

役割:

- 会話の中の「修正」だけを抜く
- 1回だけの気分をそのまま標準にしない
- 繰り返し出た修正を `PreferenceRule` へ昇格する

保存されるもの:

- `ConversationTurn`
- `PreferenceRule`

例えば

- `情報量が多すぎる`
- `余白を作れ`
- `ノリで作るな`

のような指摘が複数回出ると、次回から prompt に返る。

### 5. `PseudoIntelligenceService`

ファイル: [service.py](/Users/setohirokazu/Desktop/claude_pseudo_intelligence_core/src/claude_pseudo_intelligence/service.py)

外から使う入口です。

主なメソッド:

- `save_episode(...)`
- `save_correction(...)`
- `save_conversation_turn(...)`
- `list_entries()`
- `list_conversation_turns()`
- `list_preference_rules()`
- `find_entry(...)`
- `get_relevant_corrections(...)`
- `get_relevant_preference_rules(...)`
- `get_relevant_conversation_corrections(...)`
- `build_conversation_guidance(...)`
- `build_guidance_context(...)`
- `save_secret(...)`
- `load_secret(...)`
- `clear_secret(...)`

## どう使うか

流れは単純です。

1. Claude がタスクを受ける
2. このコアから `guidance context` を引く
3. その文脈を prompt に差し込んで Claude に考えさせる
4. 出力後に `save_episode()` で記録する
5. 人間が直したら `save_correction()` で補正を残す
6. 会話で出た修正を `save_conversation_turn()` で残す
7. 次回は案件補正と会話由来ルールの両方が返ってくる

## Claude 側の最小統合イメージ

```python
from claude_pseudo_intelligence import PseudoIntelligenceService

service = PseudoIntelligenceService("/path/to/memory")

guidance = service.build_guidance_context(
    company_profile={"basic": {"industry": "IT・情報処理・ソフトウェア"}},
    task_title="自治体システム移行支援業務",
    issuer="東京都",
    raw_text=source_text,
)

prompt = f"""
You are a proposal analyst.

{guidance}

# SOURCE TEXT
{source_text}
"""
```

タスク完了後:

```python
entry = service.save_episode(
    title="自治体システム移行支援業務",
    issuer="東京都",
    task_type="tender",
    source_text=source_text,
    output=result_payload,
)
```

人間の修正後:

```python
service.save_correction(
    entry.id,
    decision_override="条件付き参加",
    correction_note="価格競争より移行体制を前面に出すべきだった",
    reuse_note="運用定着を先置きする",
)
```

会話修正後:

```python
service.save_conversation_turn(
    task_scope="service design",
    user_message="提案サービスのトップを作れ",
    assistant_message="情報を詰め込みすぎたトップを出した",
    user_feedback="情報量が多すぎる。余白を作れ。ノリで作るな。",
)
```

## 何が嬉しいか

### 1. モデル依存を下げられる

Claude でも ChatGPT でも、外部記憶層は共通にできる。

### 2. 人間の修正が資産になる

直した内容が、次回の判断や骨子に返る。

### 3. 大量データが要らない

全文ログを無限に抱えず、

- episode
- correction
- conversation turn
- preference rule
- guidance

だけを持てばいい。

### 4. 学習の理由が追える

なぜ今回こう出したかを、過去補正まで辿れる。

## 向いている用途

- 入札判断
- 提案骨子生成
- 調査業務
- 社内ナレッジ補正
- レビュー支援
- 案件ごとの手直しが頻繁に起きる業務

## 向いていない用途

- 長大な自律エージェント全体の制御
- ベンチ分岐や `worktree` 実験そのもの
- UI を含む完成アプリ
- 重い検索基盤やベクトルDB前提の大規模知識運用

## 現在の制約

- 類似判定はまだ軽量なキーワードベース
- correction の質は、人間の記録内容に依存する
- 会話修正の抽出は軽量ヒューリスティックなので、完璧な意味理解ではない
- domain ごとの専門ルールは別途 playbook 層が必要

## 今後の拡張候補

- MCP server 化
- Claude / ChatGPT 共通 adapter
- correction の重み付け
- 成功率や採用率と結びつけた skill 昇格
- domain 別 playbook merge

## まとめ

このフォルダは、Claude に長い心得を読ませるためのものではありません。

**Claude が使う、外部の学習記憶コア**です。

やっていることは単純です。

- 仕事を残す
- 直しを残す
- 次に返す

これだけです。  
だが、この3つがあると AI は「その場しのぎ」から抜け始めます。
