"""Infer a 0.0-1.0 quality score from the user's reaction to an AI output.

Scoring priority (highest wins):
  1. Strong rejection markers        → 0.10
  2. Strong praise markers           → 0.90
  3. Mild praise / agreement         → 0.80
  4. Direction change (not rejection) → 0.65
  5. Corrections + negative tone     → 0.25
  6. Corrections + neutral/positive  → 0.60  ← learning, not failure
  7. Mild feedback without signal    → 0.50
  8. Silence                         → 0.75

Key insight: extracted_corrections often represent "learnings from insight"
rather than "the AI was wrong". User tone in feedback determines the actual
quality signal. "すごいぞ！感動した！" with corrections = 0.90, not 0.25.

This module is the ENTRY POINT of the entire correction system.
If it fails to detect feedback, nothing downstream works.
Every marker must be justified by real user behavior.
"""

from __future__ import annotations

import re

# ============================================================
# PRAISE MARKERS — 日本語の褒め・承認・同意パターンを網羅
# ============================================================
PRAISE_MARKERS = (
    # --- 直接的な称賛 ---
    "いい", "いいね", "いいぞ", "いいじゃん", "いいじゃないか",
    "よかった", "よかったよ", "よくやった",
    "完璧", "最高", "すごい", "すげえ", "すげー",
    "素晴らしい", "素敵", "見事",
    "やるじゃん", "やるね", "やるな", "やるやん",
    "さすが", "さすがだ", "さすがだな",
    "天才", "神", "化け物",
    "エクセレント", "ナイス", "グッド",
    "めっちゃいい", "めちゃくちゃいい", "めちゃいい",
    "かなりいい", "相当いい", "結構いい",
    "負ける気がしない", "すごくない",
    "かっこいい", "かわいい", "美しい",
    "センスある", "センスいい",
    "的確", "正確", "的を射", "ドンピシャ",
    "うまい", "うまいぞ", "うまいね",
    "見やすい", "わかりやすい", "使いやすい",

    # --- 感動・驚嘆 ---
    "感動", "鳥肌", "震え", "痺れ", "惚れ",
    "やばい", "やべえ", "やばすぎ",
    "半端ない", "はんぱない", "半端じゃない",
    "圧倒的", "別格", "異次元",
    "えぐい", "えぐ",

    # --- 同意・承認 ---
    "そうだね", "そうだな", "そうそう", "そう", "うん",
    "それだ", "それそれ", "それ", "これだ", "これこれ", "これだよ",
    "合ってる", "間違いない", "その通り", "正解",
    "ぴったり", "好き", "気に入った",
    "悪くない", "悪くないね", "嫌いじゃない",

    # --- 承認の軽い表現 ---
    "よし", "おし", "よしよし",
    "おけ", "おっけ", "おっけー", "おk",
    "了解", "りょ", "りょうかい", "把握",
    "ok", "OK", "オッケー",

    # --- 委任・信頼 ---
    "頼む", "頼むぞ", "頼むね", "頼んだ",
    "任せ", "まかせ", "任せた", "まかせた",
    "信じる", "信じてる", "信頼",

    # --- ポジティブな評価（方向指示はDIRECTION_MARKERSに移動済み） ---
    "そのまま", "その調子", "いい感じ",

    # --- おお・おー系（感嘆） ---
    "おお", "おー", "おおお", "おおー",
    "ほう", "ほー", "ほほう",
    "なるほど", "たしかに", "確かに",
    "まじか", "マジか", "マジで",
    "わお", "わーお",

    # --- 関西弁 ---
    "ええやん", "ええね", "ええな", "ええぞ",
    "めっちゃええ",

    # --- English: direct praise ---
    "perfect", "great", "nice", "good", "excellent", "exactly",
    "love it", "awesome", "amazing", "brilliant", "fantastic",
    "well done", "impressive", "let's go", "ship it", "nailed it",
    "wonderful", "superb", "magnificent", "stellar", "flawless",
    "top notch", "first rate", "world class",
    "beautiful", "elegant", "clean", "crisp", "solid",
    "genius", "legendary", "god tier", "goated",
    "chef's kiss", "muah",

    # --- English: casual approval ---
    "cool", "sweet", "sick", "fire", "lit", "dope", "rad", "legit",
    "clutch", "goat", "based", "bussin", "slaps",
    "yes", "yep", "yeah", "yup", "ye", "ya", "yas",
    "sure", "alright", "aight", "bet", "word",
    "that's it", "this is it", "bingo", "spot on", "bull's eye",
    "on point", "on the money", "on the nose",

    # --- English: gratitude ---
    "thank you", "thanks", "thx", "ty", "tyvm", "appreciate it",
    "grateful", "thanks a lot", "thanks so much", "much appreciated",

    # --- English: agreement ---
    "makes sense", "fair enough", "i agree", "agreed", "totally",
    "absolutely", "definitely", "for sure", "100%", "hundred percent",
    "right", "correct", "true", "exactly right", "precisely",
    "you're right", "that's right", "that's correct",

    # --- English: mild positive ---
    "not bad", "decent", "reasonable", "acceptable", "works for me",
    "i like it", "i love it", "i dig it", "this works",
    "looks good", "looks great", "lgtm",
    "keep going", "keep it up", "carry on",
    "fine", "fair", "ok", "okay",
)

STRONG_PRAISE_MARKERS = (
    # --- 日本語 ---
    "完璧", "最高", "すごい", "すげえ", "すげー",
    "感動", "素晴らしい", "見事",
    "エクセレント",
    "天才", "神", "化け物",
    "めっちゃいい", "めちゃくちゃいい",
    "負ける気がしない",
    "世界のどこにもない",
    "これ強い", "強いぞ", "強すぎ",
    "やばい", "やべえ", "やばすぎ",
    "半端ない", "はんぱない",
    "鳥肌", "震え", "痺れ", "惚れ",
    "圧倒的", "別格", "異次元",
    "えぐい", "えぐ",
    "人間超えてる", "人間じゃない",
    "最強",
    # --- English ---
    "perfect", "excellent", "amazing", "brilliant", "impressive",
    "nailed it", "incredible", "phenomenal", "outstanding",
    "masterpiece", "mind-blowing", "mind blowing",
    "genius", "legendary", "god tier", "goated",
    "insane", "unreal", "unbelievable",
    "best ever", "best i've seen", "blew my mind",
    "holy shit", "holy cow", "oh my god", "omg",
    "flawless", "top notch", "world class",
    "game changer", "next level",
    "this is incredible", "this is perfect",
    "chef's kiss",
)

# ============================================================
# NEGATIVE MARKERS — 怒り・不満・落胆・禁止パターンを網羅
# ============================================================
NEGATIVE_MARKERS = (
    # --- 否定・拒絶 ---
    "違う", "ちがう", "ちげえ", "ちげーよ",
    "そうじゃない", "そうじゃなくて", "そういうことじゃない",
    "それじゃない", "そこじゃない", "そっちじゃない",
    "いや", "いやいや", "いやいやいや",

    # --- 品質への不満 ---
    "全然", "全く", "まったく",
    "雑", "雑すぎ", "雑だ",
    "抽象的", "具体性がない", "ふわふわ", "薄い", "浅い",
    "中途半端", "不十分", "足りない", "足りてない",
    "手抜き", "適当", "テキトー", "てきとう",
    "微妙", "ビミョー", "びみょう",
    "ひどい", "ひどすぎ", "酷い",
    "最悪", "ワースト",

    # --- やり直し・修正要求 ---
    "やり直し", "やり直せ", "やりなおし", "やりなおせ",
    "作り直し", "作り直せ",
    "直せ", "直して", "修正しろ", "修正して",
    "書き直せ", "書き直して",

    # --- 嘘・不正確 ---
    "嘘", "うそ", "嘘つ", "うそつ",
    "間違", "まちが",
    "デタラメ", "でたらめ", "出鱈目",
    "正確じゃない", "正しくない",
    "推測で", "想像で", "適当な数字",

    # --- 故障・問題 ---
    "壊れ", "壊した",
    "バグ", "エラー", "動かない", "動かん",
    "落ちた", "落ちる", "クラッシュ",

    # --- 怒り・苛立ち ---
    "ふざけ", "ふざけんな", "ふざけるな",
    "ありえない", "ありえん",
    "冗談じゃない", "冗談きつい",
    "言語道断",
    "なにしてん", "何してん", "何やってん",
    "何回言えば", "何度言えば", "何度も", "毎回",
    "いい加減", "いいかげん",
    "子供か", "小学生か", "素人か",
    "勘弁", "勘弁して", "かんべん",
    "もういい", "もうええわ", "もうええ",
    "はぁ",

    # --- ダメ・使えない ---
    "ダメ", "だめ", "ダメだ", "だめだ",
    "使えない", "使えん", "使い物にならない",
    "意味ない", "意味がない", "無駄",
    "要らない", "いらない",
    "遅い", "遅いな", "遅すぎ",

    # --- しょうもない系 ---
    "しょうもない", "しょうもな",
    "くだらない", "くだらん",
    "つまらない", "つまらん",
    "ゴミ", "クソ", "くそ", "カス",
    "ポンコツ",

    # --- 命令形での修正指示（怒りのニュアンス） ---
    "ちゃんとしろ", "ちゃんとやれ", "ちゃんと見ろ", "ちゃんと読め",
    "ちゃんと確認しろ", "ちゃんと調べろ",
    "精査しろ", "精査してから",
    "確認してから", "見てから", "読んでから",
    "考えてから", "考えろ",
    "出すな", "書くな", "言うな",

    # --- 失望 ---
    "期待はずれ", "期待外れ",
    "がっかり", "ガッカリ",
    "残念", "残念だ",
    "落胆",

    # --- 疲弊 ---
    "疲れた", "疲れる", "めんどくさ",
    "もう無理", "手に負えない",
    "話にならない", "お話にならない",

    # --- English: rejection ---
    "no", "nah", "nope", "nuh uh", "hell no", "absolutely not",
    "wrong", "incorrect", "inaccurate", "false", "not true",
    "that's wrong", "that's not right", "that's incorrect",
    "not even close", "way off", "missed the mark",

    # --- English: quality complaints ---
    "terrible", "awful", "horrible", "atrocious", "abysmal",
    "bad", "poor", "weak", "lazy", "sloppy", "careless",
    "half-assed", "half assed", "low effort", "low quality",
    "mediocre", "subpar", "below average",
    "broken", "buggy", "doesn't work", "not working",
    "vague", "unclear", "confusing", "nonsense", "gibberish",
    "too long", "too short", "too verbose", "too abstract",
    "generic", "boilerplate", "cookie cutter", "copy paste",
    "shallow", "superficial", "surface level",
    "incomplete", "missing", "forgot", "left out", "skipped",
    "outdated", "stale", "irrelevant", "off topic",

    # --- English: redo requests ---
    "fix it", "fix this", "redo it", "redo this", "try again",
    "start over", "do it again", "one more time",
    "rewrite", "rewrite this", "rewrite it",
    "not what i asked", "not what i wanted", "not what i meant",
    "i said", "i already said", "i told you", "i literally said",
    "read it again", "look again", "check again",
    "did you even read", "did you even look",

    # --- English: frustration ---
    "wtf", "wth", "come on", "seriously", "really",
    "for real", "are you kidding", "you're kidding",
    "i can't believe", "unbelievable",
    "this is ridiculous", "this is absurd",
    "what the hell", "what is this", "what are you doing",
    "how many times", "i keep telling you", "again and again",
    "ugh", "smh", "ffs", "jfc", "bruh",

    # --- English: dismissal ---
    "useless", "pointless", "garbage", "trash", "rubbish", "junk",
    "waste of time", "not helpful", "unhelpful",
    "disappointing", "disappointed", "let down",
    "frustrated", "frustrating", "annoying", "annoyed",
    "stop", "don't", "enough", "never mind", "forget it",
    "i'll do it myself", "i'll just do it",
)

STRONG_REJECTION_MARKERS = (
    # --- 日本語 ---
    "全然違う", "全く違う",
    "やり直し", "やり直せ", "作り直し", "作り直せ",
    "ひどい", "ひどすぎ", "酷い", "酷すぎ",
    "最悪", "ワースト",
    "使えない", "使い物にならない",
    "嘘ばっか", "嘘つくな", "うそつけ", "デタラメ",
    "ふざけんな", "ふざけるな",
    "ありえない", "ありえん",
    "言語道断",
    "冗談じゃない", "冗談きつい",
    "話にならない", "お話にならない",
    "もう無理", "手に負えない",
    "子供か", "小学生か", "素人か",
    "何回言えば", "何度言えば",
    "いい加減にしろ", "いいかげんにしろ",
    "ゴミ", "クソ", "カス", "ポンコツ",
    "しょうもない",
    # --- English ---
    "terrible", "awful", "horrible", "atrocious", "abysmal",
    "wtf", "wth", "ffs", "jfc",
    "useless", "garbage", "trash", "rubbish", "junk",
    "not what i asked", "not what i wanted", "this is wrong",
    "completely wrong", "totally wrong", "way off",
    "start over", "do it again",
    "how many times", "i keep telling you",
    "waste of time", "i'll do it myself",
    "are you kidding", "you're kidding",
    "this is ridiculous", "this is absurd",
    "did you even read", "did you even look",
    "what the hell", "what is this",
    "i can't believe",
    "worst", "the worst",
)

# ============================================================
# DIRECTION MARKERS — 方向転換（拒絶ではなくステアリング）
# ============================================================
DIRECTION_MARKERS = (
    # --- 次の指示 ---
    "じゃあ", "では", "それじゃ",
    "次は", "次に", "次いこう",
    "それを", "これを",
    "こっち", "こっちの", "あっちの",

    # --- 作業指示 ---
    "やってみ", "やってみて", "やってみよう",
    "作れ", "作って", "作るんだ",
    "実装しろ", "実装して",
    "調べろ", "調べて",
    "試して", "試しに", "試してみ",
    "見せて", "見して",
    "教えて",

    # --- 深掘り ---
    "もっと", "深く", "掘れ", "掘って",
    "詳しく", "詳細に",
    "広げて", "展開して",

    # --- 継続指示 ---
    "続けろ", "続けて", "続き",
    "進め", "進めて", "進めよう",
    "行け", "行こう",
    "やってくれ", "やって",
    "頼む", "まかせ", "任せ",
    "お願い", "おねがい",

    # --- 方向確認 ---
    "それでいい", "それでいこう", "それで行こう",
    "その方向", "その方向で", "その線で",
    "そのまま", "そのままで",
    "あとは", "ついでに",
    "ところで", "そういえば",
    "他にも", "他には",

    # --- English: direction ---
    "then", "next", "now", "moving on", "let's move on",
    "go ahead", "go for it", "proceed", "continue",
    "try", "check", "look into", "explore", "investigate",
    "also", "and then", "after that", "once that's done",
    "what about", "how about", "what if", "could you",
    "can you", "would you", "please",
    "let's try", "let's see", "let's do",
    "instead", "rather", "alternatively",
    "actually", "on second thought", "wait",
    "one more thing", "by the way", "btw",
    "that's fine", "that works", "good enough",
    "let's go with", "go with that", "sounds good",
    "do it", "make it", "build it", "run it", "ship it",
)

# ============================================================
# REGEX PATTERNS — 動詞活用形での感情検出
# ============================================================

# 禁止形: ～するな、～出すな、～書くな etc.
_RE_PROHIBITION = re.compile(
    r'[るすくつぬぶむえけげせてねべめれ]な(?![いくけさし])'
    r'|しないで|やめろ|やめて|やめい'
    r'|するな|出すな|書くな|言うな|使うな|入れるな|消すな|変えるな'
    r'|触るな|いじるな|弄るな',
    re.IGNORECASE,
)

# 強い命令形（修正要求のニュアンス）: ちゃんと～しろ、～してから
_RE_CORRECTION_COMMAND = re.compile(
    r'ちゃんと.{0,8}[しやせ][ろれてえ]'
    r'|してから[言書出話説進]'
    r'|確認してから|見てから|読んでから|調べてから|精査してから'
    r'|考えてから|試してから',
    re.IGNORECASE,
)

# 疑問形の苛立ち: なんで～してんの、何回言えば
_RE_FRUSTRATION_QUESTION = re.compile(
    r'なんで.{0,15}[てし](?:る|た|ん)'
    r'|なぜ.{0,15}[てし](?:る|た|ん)'
    r'|どうして.{0,15}[てし](?:る|た|ん)'
    r'|何回[言聞]えば'
    r'|何度[言聞]えば'
    r'|何回目'
    r'|わかってる\?|分かってる\?|わかってんの|聞いてる\?',
    re.IGNORECASE,
)

# ネガティブ強調: めちゃくちゃ＋ネガティブワード
_RE_INTENSIFIED_NEGATIVE = re.compile(
    r'(?:めちゃくちゃ|めっちゃ|超|クソ|くそ|激|鬼|死ぬほど)'
    r'.{0,6}'
    r'(?:ダメ|だめ|ひどい|酷い|遅い|雑|薄い|浅い|微妙|使えない|わからん|わかりにくい)',
    re.IGNORECASE,
)


# English: "don't ...", "stop ...-ing", "never ..."
_RE_EN_PROHIBITION = re.compile(
    r"\bdon'?t\s+\w+"
    r"|\bstop\s+\w+ing"
    r"|\bnever\s+\w+"
    r"|\bquit\s+\w+ing",
    re.IGNORECASE,
)

# English: "you should have", "why didn't you", "I told you to"
_RE_EN_FRUSTRATION = re.compile(
    r"\byou\s+should\s+have\b"
    r"|\bwhy\s+didn'?t\s+you\b"
    r"|\bI\s+(?:already\s+)?told\s+you\b"
    r"|\bhow\s+many\s+times\b"
    r"|\bi\s+(?:already\s+)?said\b"
    r"|\bI\s+keep\s+(?:telling|asking|saying)\b"
    r"|\bdid\s+you\s+even\s+(?:read|look|listen|try)\b",
    re.IGNORECASE,
)

# English: intensifier + negative ("so bad", "really awful", "absolutely terrible")
_RE_EN_INTENSIFIED_NEGATIVE = re.compile(
    r"\b(?:so|really|very|extremely|absolutely|completely|totally|utterly|incredibly)\s+"
    r"(?:bad|awful|terrible|horrible|wrong|broken|useless|annoying|frustrating|disappointing|confusing|slow|ugly)\b",
    re.IGNORECASE,
)


def _check_regex_negative(text: str) -> bool:
    """Check if any regex-based negative patterns match."""
    return bool(
        _RE_PROHIBITION.search(text)
        or _RE_CORRECTION_COMMAND.search(text)
        or _RE_FRUSTRATION_QUESTION.search(text)
        or _RE_INTENSIFIED_NEGATIVE.search(text)
        or _RE_EN_PROHIBITION.search(text)
        or _RE_EN_FRUSTRATION.search(text)
        or _RE_EN_INTENSIFIED_NEGATIVE.search(text)
    )


def _check_regex_strong_negative(text: str) -> bool:
    """Check if strong regex-based negative patterns match."""
    return bool(
        _RE_FRUSTRATION_QUESTION.search(text)
        or _RE_INTENSIFIED_NEGATIVE.search(text)
        or _RE_EN_FRUSTRATION.search(text)
        or _RE_EN_INTENSIFIED_NEGATIVE.search(text)
    )


def score_reaction(
    user_feedback: str,
    extracted_corrections: list[str],
) -> float:
    """
    Infer quality score from user's reaction.

    Priority: user tone > behavioral signals > correction presence.
    Corrections with praise = learning (high score).
    Corrections with rejection = actual failure (low score).
    No emotional markers = behavioral default (0.65 for corrections, 0.75 for silence).

    Designed for users who DO express emotion AND users who DON'T.
    Emotionally flat users get moderate scores (not penalized for silence),
    and corrections without negativity default to "learning" (0.60-0.65).

    Args:
        user_feedback:        Raw user feedback text (may be empty).
        extracted_corrections: Correction candidates already extracted from the turn.

    Returns:
        float in [0.0, 1.0]
    """
    feedback = (user_feedback or "").strip().lower()
    has_corrections = bool(extracted_corrections)

    # Silence = silent approval
    if not feedback and not has_corrections:
        return 0.75

    # Strip negation prefixes that flip sentiment ("not bad" → praise, not negative)
    _negation_flips = re.compile(
        r"\bnot\s+(?:bad|awful|terrible|horrible|wrong|the worst)\b"
        r"|\bnot\s+(?:half\s+)?bad\b"
        r"|\b(?:isn't|ain't|wasn't)\s+(?:bad|awful|terrible)\b",
        re.IGNORECASE,
    )
    feedback_for_negative = _negation_flips.sub("", feedback)

    # Detect tone signals (marker-based)
    has_strong_rejection = any(marker in feedback_for_negative for marker in STRONG_REJECTION_MARKERS)
    has_negative = any(marker in feedback_for_negative for marker in NEGATIVE_MARKERS)
    has_strong_praise = any(marker in feedback for marker in STRONG_PRAISE_MARKERS)
    has_praise = any(marker in feedback for marker in PRAISE_MARKERS)
    has_direction = any(marker in feedback for marker in DIRECTION_MARKERS)

    # Regex-based detection (catches verb conjugations, prohibition forms, etc.)
    has_regex_negative = _check_regex_negative(feedback)
    has_regex_strong_negative = _check_regex_strong_negative(feedback)

    # Merge regex results into marker flags
    if has_regex_strong_negative:
        has_strong_rejection = True
    if has_regex_negative:
        has_negative = True

    has_any_emotion = has_strong_rejection or has_negative or has_strong_praise or has_praise

    # 1. Strong rejection always wins
    if has_strong_rejection:
        return 0.10

    # 2. Strong praise always wins (even with corrections = "learning moment")
    if has_strong_praise:
        return 0.90

    # 3. Mild praise (even with corrections)
    if has_praise and not has_negative:
        return 0.80 if has_corrections else 0.85

    # 4. Direction change without negativity = steering, not failure
    if has_direction and not has_negative:
        return 0.65

    # 5. Negative tone + corrections = actual failure
    if has_negative and has_corrections:
        return 0.25

    # 6. Negative tone without corrections = mild dissatisfaction
    if has_negative:
        return 0.30

    # 7. Corrections present, no emotional markers = neutral learning
    #    (Not a failure. The user provided new insight without judging quality.)
    if has_corrections and not has_any_emotion:
        return 0.65

    # 8. Corrections present with some non-negative signal
    if has_corrections:
        return 0.60

    # 9. Feedback exists but no clear signal (emotionally flat)
    if feedback:
        return 0.55

    # 10. Fallback
    return 0.75


def reaction_label(score: float) -> str:
    """Human-readable label for a reaction score."""
    if score >= 0.85:
        return "praised"
    if score >= 0.65:
        return "accepted"
    if score >= 0.35:
        return "mixed"
    return "corrected"
