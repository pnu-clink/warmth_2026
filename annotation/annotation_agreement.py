#!/usr/bin/env python3
"""세 명의 Human annotation CSV 간 명목척도 일치도를 계산한다.

외부 패키지 없이 Python 표준 라이브러리만 사용한다. 문항은 행 순서가 아니라
ID 열로 정렬하며, 빈 라벨과 누락 문항은 결측치로 처리한다.

기본 실행 예:
    python annotation_agreement.py annotator1.csv annotator2.csv annotator3.csv

열 이름을 직접 지정하는 예:
    python annotation_agreement.py a.csv b.csv c.csv \
        --id-column item_id --label-column error_category \
        --output-prefix task2_midcheck
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


AUTO_ID_COLUMNS = ("item_id", "question_id", "id")
AUTO_LABEL_COLUMNS = ("knowledge_type", "error_category", "label")
METADATA_COLUMNS = ("question_id", "category", "question")
ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp949")


@dataclass
class AnnotationFile:
    path: Path
    name: str
    encoding: str
    fieldnames: List[str]
    rows: Dict[str, Dict[str, str]]


def normalize_text(value: object, case_insensitive: bool = False) -> str:
    """비교용 문자열을 NFC 정규화하고 앞뒤/연속 공백을 정리한다."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold() if case_insensitive else text


def natural_key(value: str) -> Tuple[object, ...]:
    """E2가 E10보다 먼저 오도록 ID를 자연 정렬한다."""
    return tuple(int(part) if part.isdigit() else part.casefold()
                 for part in re.split(r"(\d+)", value))


def read_csv_with_fallback(path: Path) -> Tuple[str, List[str], List[Dict[str, str]]]:
    last_error: Optional[Exception] = None
    for encoding in ENCODING_CANDIDATES:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError("헤더가 없습니다.")
                fieldnames = [normalize_text(name) for name in reader.fieldnames]
                if len(set(fieldnames)) != len(fieldnames):
                    raise ValueError("정규화 후 중복되는 열 이름이 있습니다.")
                rows = []
                for raw in reader:
                    rows.append({fieldnames[index]: value
                                 for index, value in enumerate(raw.values())
                                 if index < len(fieldnames)})
                return encoding, fieldnames, rows
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(
        f"인코딩을 판별하지 못했습니다(시도: {', '.join(ENCODING_CANDIDATES)}): {last_error}"
    )


def unique_annotator_names(paths: Sequence[Path]) -> List[str]:
    base_names = [path.stem for path in paths]
    counts = Counter(base_names)
    seen: Counter[str] = Counter()
    names: List[str] = []
    for name in base_names:
        seen[name] += 1
        names.append(name if counts[name] == 1 else f"{name}_{seen[name]}")
    return names


def choose_common_column(
    fieldsets: Sequence[set[str]], explicit: Optional[str], candidates: Sequence[str], kind: str
) -> str:
    if explicit:
        missing = [index + 1 for index, fields in enumerate(fieldsets) if explicit not in fields]
        if missing:
            raise ValueError(f"{kind} 열 '{explicit}'이(가) 없는 파일: {missing}")
        return explicit
    for candidate in candidates:
        if all(candidate in fields for fields in fieldsets):
            return candidate
    raise ValueError(
        f"세 파일에 공통인 {kind} 열을 자동 탐지하지 못했습니다. "
        f"--{'id-column' if kind == 'ID' else 'label-column'}으로 지정하세요."
    )


def load_annotations(
    paths: Sequence[Path], id_column: Optional[str], label_column: Optional[str]
) -> Tuple[List[AnnotationFile], str, str]:
    if len(paths) != 3:
        raise ValueError("CSV 파일을 정확히 3개 입력해야 합니다.")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"파일을 찾을 수 없습니다: {path}")

    raw_files = [read_csv_with_fallback(path) for path in paths]
    fieldsets = [set(fieldnames) for _, fieldnames, _ in raw_files]
    chosen_id = choose_common_column(fieldsets, id_column, AUTO_ID_COLUMNS, "ID")
    chosen_label = choose_common_column(fieldsets, label_column, AUTO_LABEL_COLUMNS, "라벨")
    names = unique_annotator_names(paths)

    files: List[AnnotationFile] = []
    for path, name, (encoding, fieldnames, raw_rows) in zip(paths, names, raw_files):
        indexed: Dict[str, Dict[str, str]] = {}
        for row_number, row in enumerate(raw_rows, start=2):
            item_id = normalize_text(row.get(chosen_id))
            if not item_id:
                raise ValueError(f"{path.name} {row_number}행: '{chosen_id}' 값이 비어 있습니다.")
            if item_id in indexed:
                raise ValueError(f"{path.name}: 중복 ID '{item_id}'이(가) 있습니다.")
            indexed[item_id] = row
        files.append(AnnotationFile(path, name, encoding, fieldnames, indexed))
    return files, chosen_id, chosen_label


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Tuple[Optional[float], float]:
    if len(labels_a) != len(labels_b) or not labels_a:
        return None, math.nan
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    categories = set(count_a) | set(count_b)
    expected = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)
    if math.isclose(expected, 1.0):
        return None, observed
    return (observed - expected) / (1.0 - expected), observed


def fleiss_kappa(complete_ratings: Sequence[Sequence[str]]) -> Optional[float]:
    """모든 평가자가 응답한 문항에 대해 3인 Fleiss' kappa를 계산한다."""
    if not complete_ratings:
        return None
    n_raters = len(complete_ratings[0])
    if n_raters < 2 or any(len(row) != n_raters for row in complete_ratings):
        return None
    category_totals: Counter[str] = Counter()
    item_agreements: List[float] = []
    for ratings in complete_ratings:
        counts = Counter(ratings)
        category_totals.update(ratings)
        numerator = sum(count * count for count in counts.values()) - n_raters
        item_agreements.append(numerator / (n_raters * (n_raters - 1)))
    p_bar = sum(item_agreements) / len(item_agreements)
    total_ratings = len(complete_ratings) * n_raters
    p_expected = sum((count / total_ratings) ** 2 for count in category_totals.values())
    if math.isclose(p_expected, 1.0):
        return None
    return (p_bar - p_expected) / (1.0 - p_expected)


def krippendorff_alpha_nominal(ratings_by_item: Sequence[Sequence[str]]) -> Optional[float]:
    """결측치를 허용하는 명목척도 Krippendorff's alpha를 계산한다."""
    coincidences: Dict[str, Counter[str]] = defaultdict(Counter)
    coincidence_total = 0.0
    for ratings in ratings_by_item:
        present = [label for label in ratings if label]
        count = len(present)
        if count < 2:
            continue
        weight = 1.0 / (count - 1)
        for i, first in enumerate(present):
            for j, second in enumerate(present):
                if i != j:
                    coincidences[first][second] += weight
        coincidence_total += count

    if coincidence_total <= 1:
        return None
    observed_disagreement = sum(
        value
        for first, row in coincidences.items()
        for second, value in row.items()
        if first != second
    ) / coincidence_total
    marginals = {
        category: sum(row.values()) for category, row in coincidences.items()
    }
    expected_disagreement = (
        sum(first_count * second_count
            for first, first_count in marginals.items()
            for second, second_count in marginals.items()
            if first != second)
        / (coincidence_total * (coincidence_total - 1))
    )
    if math.isclose(expected_disagreement, 0.0):
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def format_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.6f}"
    return str(value)


def write_dict_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def analyze(
    files: Sequence[AnnotationFile], id_column: str, label_column: str,
    output_prefix: Path, case_insensitive: bool
) -> Tuple[Path, Path, Path]:
    all_ids = sorted(set().union(*(file.rows for file in files)), key=natural_key)
    id_sets = [set(file.rows) for file in files]
    common_ids = set.intersection(*id_sets)

    comparison_rows: List[Dict[str, object]] = []
    ratings_by_item: List[List[str]] = []
    complete_ratings: List[List[str]] = []
    label_counts = [Counter() for _ in files]
    three_way_agree = 0

    for item_id in all_ids:
        labels: List[str] = []
        raw_labels: List[str] = []
        present_flags: List[bool] = []
        metadata: Dict[str, str] = {}
        for file_index, file in enumerate(files):
            row = file.rows.get(item_id)
            present_flags.append(row is not None)
            raw_label = normalize_text(row.get(label_column)) if row else ""
            label = normalize_text(raw_label, case_insensitive=case_insensitive)
            raw_labels.append(raw_label)
            labels.append(label)
            if label:
                label_counts[file_index][raw_label] += 1
            if row:
                for column in METADATA_COLUMNS:
                    if column != id_column and column not in metadata:
                        value = normalize_text(row.get(column))
                        if value:
                            metadata[column] = value

        ratings_by_item.append(labels)
        rated_count = sum(bool(label) for label in labels)
        is_complete = rated_count == 3
        all_agree = is_complete and len(set(labels)) == 1
        if is_complete:
            complete_ratings.append(labels)
            three_way_agree += int(all_agree)

        if not all(present_flags):
            status = "missing_item"
        elif not is_complete:
            status = "missing_label"
        elif all_agree:
            status = "agree"
        else:
            status = "disagree"

        nonempty_counts = Counter(label for label in raw_labels if label)
        majority = ""
        if nonempty_counts:
            top_label, top_count = nonempty_counts.most_common(1)[0]
            if top_count >= 2:
                majority = top_label
        if all_agree:
            pattern = "all_same"
        elif not is_complete:
            pattern = "incomplete"
        elif len(set(labels)) == 3:
            pattern = "all_different"
        else:
            pattern = "two_vs_one"

        output_row: Dict[str, object] = {
            id_column: item_id,
            "status": status,
            "agreement_pattern": pattern,
            "rated_count": rated_count,
            "majority_label": majority,
        }
        for column in METADATA_COLUMNS:
            if column != id_column:
                output_row[column] = metadata.get(column, "")
        for index, (file, raw_label, present) in enumerate(
            zip(files, raw_labels, present_flags), start=1
        ):
            output_row[f"annotator_{index}"] = file.name
            output_row[f"label_{index}"] = raw_label
            output_row[f"item_present_{index}"] = "Y" if present else "N"
        comparison_rows.append(output_row)

    complete_count = len(complete_ratings)
    exact_rate = safe_ratio(three_way_agree, complete_count)
    fleiss = fleiss_kappa(complete_ratings)
    alpha = krippendorff_alpha_nominal(ratings_by_item)

    summary_rows: List[Dict[str, object]] = []

    def add(section: str, metric: str, value: object, n_items: object = "", note: str = "") -> None:
        summary_rows.append({
            "section": section, "metric": metric, "value": value,
            "n_items": n_items, "note": note,
        })

    add("setting", "id_column", id_column)
    add("setting", "label_column", label_column)
    add("coverage", "union_items", len(all_ids), len(all_ids), "세 파일 중 하나 이상에 존재")
    add("coverage", "common_item_ids", len(common_ids), len(common_ids), "세 파일 모두에 ID 존재")
    add("coverage", "complete_case_items", complete_count, complete_count, "세 라벨이 모두 비어 있지 않음")
    for index, (file, counts) in enumerate(zip(files, label_counts), start=1):
        rated = sum(counts.values())
        add("annotator", f"annotator_{index}_name", file.name, rated, f"encoding={file.encoding}")
        add("annotator", f"annotator_{index}_rated_items", rated, rated)
    add("overall", "three_way_exact_agreement_count", three_way_agree, complete_count)
    add("overall", "three_way_exact_agreement_rate", exact_rate, complete_count)
    add("overall", "fleiss_kappa", fleiss, complete_count, "완전 사례만 사용")
    alpha_item_count = sum(sum(bool(label) for label in row) >= 2 for row in ratings_by_item)
    add("overall", "krippendorff_alpha_nominal", alpha, alpha_item_count, "평가가 2개 이상인 문항 사용")

    for first in range(3):
        for second in range(first + 1, 3):
            pair_a: List[str] = []
            pair_b: List[str] = []
            for ratings in ratings_by_item:
                if ratings[first] and ratings[second]:
                    pair_a.append(ratings[first])
                    pair_b.append(ratings[second])
            kappa, observed = cohen_kappa(pair_a, pair_b)
            pair_name = f"{files[first].name}__vs__{files[second].name}"
            add("pairwise", f"percent_agreement__{pair_name}", observed, len(pair_a))
            add("pairwise", f"cohen_kappa__{pair_name}", kappa, len(pair_a))

    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    disagreements_path = output_prefix.with_name(output_prefix.name + "_disagreements.csv")
    distribution_path = output_prefix.with_name(output_prefix.name + "_label_distribution.csv")
    write_dict_csv(
        summary_path, ("section", "metric", "value", "n_items", "note"), summary_rows
    )

    comparison_fields = [id_column, "question_id", "category", "question", "status",
                         "agreement_pattern", "rated_count", "majority_label"]
    for index in range(1, 4):
        comparison_fields.extend((f"annotator_{index}", f"label_{index}", f"item_present_{index}"))
    # 일치한 완전 사례는 제외하고, 검토가 필요한 행만 내보낸다.
    disagreement_rows = [row for row in comparison_rows if row["status"] != "agree"]
    write_dict_csv(disagreements_path, comparison_fields, disagreement_rows)

    all_labels = sorted(set().union(*(set(counts) for counts in label_counts)), key=natural_key)
    distribution_rows = []
    for label in all_labels:
        row: Dict[str, object] = {"label": label}
        total = 0
        for index, counts in enumerate(label_counts, start=1):
            row[f"annotator_{index}"] = counts[label]
            total += counts[label]
        row["total"] = total
        distribution_rows.append(row)
    write_dict_csv(
        distribution_path,
        ("label", "annotator_1", "annotator_2", "annotator_3", "total"),
        distribution_rows,
    )

    print("\n=== 3인 어노테이션 중간 일치도 ===")
    print(f"ID 열: {id_column} | 라벨 열: {label_column}")
    print(f"전체 ID(합집합): {len(all_ids)} | 공통 ID: {len(common_ids)} | 완전 사례: {complete_count}")
    print(f"3인 완전 일치율: {format_value(exact_rate)} ({three_way_agree}/{complete_count})")
    print(f"Fleiss' kappa: {format_value(fleiss)}")
    print(f"Krippendorff's alpha (nominal): {format_value(alpha)}")
    for row in summary_rows:
        if row["section"] == "pairwise" and str(row["metric"]).startswith("cohen_kappa"):
            print(f"{row['metric']}: {format_value(row['value'])} (n={row['n_items']})")
    print("\n출력 파일:")
    print(f"- {summary_path.resolve()}")
    print(f"- {disagreements_path.resolve()}")
    print(f"- {distribution_path.resolve()}")
    if any(row["status"] in {"missing_item", "missing_label"} for row in comparison_rows):
        print("주의: 누락 문항 또는 빈 라벨이 있습니다. disagreements 파일을 확인하세요.")
    if fleiss is None or alpha is None:
        print("주의: NA는 비교 가능한 사례 부족 또는 라벨 분산이 없어 통계가 정의되지 않은 경우입니다.")
    return summary_path, disagreements_path, distribution_path


def self_test() -> None:
    kappa, agreement = cohen_kappa(["A", "A", "B", "B"], ["A", "B", "B", "B"])
    assert math.isclose(agreement, 0.75)
    assert kappa is not None and math.isclose(kappa, 0.5)
    perfect = [["A", "A", "A"], ["B", "B", "B"], ["A", "A", "A"]]
    assert math.isclose(fleiss_kappa(perfect) or 0.0, 1.0)
    assert math.isclose(krippendorff_alpha_nominal(perfect) or 0.0, 1.0)
    missing_perfect = [["A", "A", ""], ["B", "", "B"], ["A", "A", "A"]]
    assert math.isclose(krippendorff_alpha_nominal(missing_perfect) or 0.0, 1.0)
    assert fleiss_kappa([["A", "A", "A"]]) is None
    print("Self-test passed.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3개의 annotation CSV를 ID로 정렬하여 명목척도 일치도를 계산합니다."
    )
    parser.add_argument("csv_files", nargs="*", type=Path, help="평가자별 CSV 파일 3개")
    parser.add_argument("--id-column", help="문항 ID 열(기본: item_id, question_id, id 순 자동 탐지)")
    parser.add_argument(
        "--label-column",
        help="평가 라벨 열(기본: knowledge_type, error_category, label 순 자동 탐지)",
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        help="출력 경로 접두사(기본: 첫 CSV 폴더/agreement_<라벨열>)",
    )
    parser.add_argument(
        "--case-insensitive", action="store_true",
        help="영문 대소문자를 구분하지 않고 라벨 비교",
    )
    parser.add_argument("--self-test", action="store_true", help="내장 통계 테스트만 실행")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if len(args.csv_files) != 3:
        print("오류: 평가자별 CSV 파일을 정확히 3개 입력하세요.", file=sys.stderr)
        return 2
    try:
        paths = [path.expanduser().resolve() for path in args.csv_files]
        files, id_column, label_column = load_annotations(
            paths, args.id_column, args.label_column
        )
        output_prefix = args.output_prefix
        if output_prefix is None:
            output_prefix = paths[0].parent / f"agreement_{label_column}"
        else:
            output_prefix = output_prefix.expanduser().resolve()
        analyze(files, id_column, label_column, output_prefix, args.case_insensitive)
        return 0
    except (OSError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
