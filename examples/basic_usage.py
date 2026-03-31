from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from claude_pseudo_intelligence import PseudoIntelligenceService


def main():
    service = PseudoIntelligenceService(
        ROOT_DIR / ".local-memory"
    )

    entry = service.save_episode(
        title="自治体DX提案のGo/No-Go判断",
        issuer="東京都",
        task_type="tender",
        source_text="自治体 システム 移行 運用保守 セキュリティ",
        company_profile={"basic": {"industry": "IT・情報処理・ソフトウェア", "name": "A社"}},
        output={"decision": "参加推奨"},
    )
    service.save_correction(
        entry.id,
        decision_override="条件付き参加",
        correction_note="価格競争より移行体制を前面に出すべきだった",
        reuse_note="運用定着と引継ぎ体制を先に見せる",
    )
    service.save_conversation_turn(
        task_scope="service design",
        user_message="提案サービスのトップ画面を作れ",
        assistant_message="情報を詰め込みすぎたトップを出した",
        user_feedback="情報量が多すぎる。余白を作れ。ノリで作るな。",
    )
    service.save_conversation_turn(
        task_scope="service design",
        user_message="もう一度トップを作れ",
        assistant_message="説明を増やした",
        user_feedback="余白を作れ。説明を削れ。",
    )

    context = service.build_guidance_context(
        company_profile={"basic": {"industry": "IT・情報処理・ソフトウェア"}},
        task_title="自治体システム移行支援業務",
        issuer="東京都",
        raw_text="システム 移行 運用保守 セキュリティ",
        task_scope="service design",
    )
    print(context)


if __name__ == "__main__":
    main()
