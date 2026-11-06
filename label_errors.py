#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""질문·선택지·정답·응답을 포함하는 전체 오류 judge 실행 진입점.

API 키: 이 파일과 같은 폴더의 .env에 OPENAI_API_KEY를 설정하세요.
입력: 검증된 MMLU-Pro / Finance / Medical의 full-context 오답 입력.
실행 없이 점검: python label_errors.py
96개 파일럿:    python label_errors.py --pilot-per-cell 2 --execute
전체/이어하기:  python label_errors.py --execute --resume
실패 재시도:    python label_errors.py --execute --resume --retry-failed

옵션 없이 실행하면 API를 호출하지 않습니다. 기존 CSV/라벨은 수정하지
않습니다. 결과는 experiments/judge_paired_20260928/judge_v2/results.jsonl에
응답마다 저장되며, 준비된 paired_tests.py에서 그대로 읽을 수 있습니다.
새 seed/체크포인트 결과는 자동 검색하지 않습니다. 별도 입력을 준비한 뒤
--input, --mapping, --output으로 지정해야 기존 결과와 섞이지 않습니다.
"""
from pathlib import Path
import sys

ENGINE_DIR = Path(__file__).resolve().parent / "experiments" / "judge_paired_20260928"


def main(argv=None):
    if not (ENGINE_DIR / "judge_full_context.py").is_file():
        raise SystemExit("Missing experiments/judge_paired_20260928/judge_full_context.py")
    # Share the validated prompt, journaling and resume logic with paired analysis.
    sys.path.insert(0, str(ENGINE_DIR))
    from judge_full_context import main as run_judge
    return run_judge(argv)


if __name__ == "__main__":
    main()
