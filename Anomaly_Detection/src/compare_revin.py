"""
LSTM_AE w/ RevIN vs w/o RevIN 결과 비교 스크립트
실행: python compare_revin.py --log_no_revin <path> --log_revin <path>
"""
import re
import argparse


# ── 파싱 헬퍼 ────────────────────────────────────────────────────────────────

def parse_test_metrics(log_path: str) -> dict:
    """로그 파일에서 TEST 결과 지표를 파싱합니다."""
    metrics = {}
    patterns = {
        'loss':      r'Loss[:\s]+([\d.]+)',
        'f1':        r'F1[:\s]+([\d.]+)',
        'precision': r'Precision[:\s]+([\d.]+)',
        'recall':    r'Recall[:\s]+([\d.]+)',
        'roc_auc':   r'ROC[_\s]?AUC[:\s]+([\d.]+)',
        'auprc':     r'AUPRC[:\s]+([\d.]+)',
    }

    in_test_section = False
    with open(log_path, 'r') as f:
        for line in f:
            if 'Start TEST Evaluation' in line:
                in_test_section = True
            if in_test_section and 'TEST Results' in line:
                for key, pat in patterns.items():
                    m = re.search(pat, line)
                    if m:
                        metrics[key] = float(m.group(1))
                break  # TEST 결과 라인 한 줄로 충분

    # fallback: TEST Results 라인이 없을 경우 개별 라인 탐색
    if not metrics:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'Start TEST Evaluation' in line:
                # 이후 30줄 내에서 지표 탐색
                for sub in lines[i:i+30]:
                    for key, pat in patterns.items():
                        if key not in metrics:
                            m = re.search(pat, sub)
                            if m:
                                metrics[key] = float(m.group(1))
                break

    return metrics


def parse_train_losses(log_path: str) -> list:
    """각 EPOCH의 평균 학습 loss를 파싱합니다."""
    losses = []
    pattern = re.compile(r'EPOCH\s+(\d+)/\d+.*Loss:[\s\d.]+\(([\d.]+)\)')
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                losses.append((int(m.group(1)), float(m.group(2))))
    return losses


# ── 출력 헬퍼 ────────────────────────────────────────────────────────────────

def print_comparison(no_revin: dict, revin: dict):
    metrics = ['loss', 'f1', 'precision', 'recall', 'roc_auc', 'auprc']
    higher_better = {'f1', 'precision', 'recall', 'roc_auc', 'auprc'}

    col_w = 14
    header = f"{'Metric':<14} {'w/o RevIN':>{col_w}} {'w/ RevIN':>{col_w}} {'Diff':>{col_w}} {'Winner':>{col_w}}"
    sep    = '-' * len(header)

    print(sep)
    print(header)
    print(sep)

    for key in metrics:
        v_no  = no_revin.get(key, None)
        v_yes = revin.get(key,    None)

        if v_no is None or v_yes is None:
            print(f"{key:<14} {'N/A':>{col_w}} {'N/A':>{col_w}} {'N/A':>{col_w}} {'N/A':>{col_w}}")
            continue

        diff = v_yes - v_no
        if key in higher_better:
            winner = '✅ w/ RevIN' if diff > 0 else ('✅ w/o RevIN' if diff < 0 else 'TIE')
        else:
            winner = '✅ w/ RevIN' if diff < 0 else ('✅ w/o RevIN' if diff > 0 else 'TIE')

        sign = '+' if diff >= 0 else ''
        print(f"{key:<14} {v_no:>{col_w}.4f} {v_yes:>{col_w}.4f} {sign+f'{diff:.4f}':>{col_w}} {winner:>{col_w}}")

    print(sep)


def print_train_curve(no_revin_losses: list, revin_losses: list):
    if not no_revin_losses and not revin_losses:
        return

    print("\n📈  Epoch별 Train Loss 비교")
    print(f"{'Epoch':<8} {'w/o RevIN':>12} {'w/ RevIN':>12}")
    print('-' * 34)

    all_epochs = sorted(set([e for e, _ in no_revin_losses] + [e for e, _ in revin_losses]))
    no_dict  = dict(no_revin_losses)
    yes_dict = dict(revin_losses)

    for ep in all_epochs:
        v_no  = f"{no_dict[ep]:.4f}"  if ep in no_dict  else 'N/A'
        v_yes = f"{yes_dict[ep]:.4f}" if ep in yes_dict else 'N/A'
        print(f"{ep:<8} {v_no:>12} {v_yes:>12}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_no_revin', required=True, help='w/o RevIN 로그 경로')
    parser.add_argument('--log_revin',    required=True, help='w/ RevIN 로그 경로')
    args = parser.parse_args()

    no_revin_metrics = parse_test_metrics(args.log_no_revin)
    revin_metrics    = parse_test_metrics(args.log_revin)
    no_revin_losses  = parse_train_losses(args.log_no_revin)
    revin_losses     = parse_train_losses(args.log_revin)

    print("\n🔬  LSTM_AE  RevIN 적용 여부 비교 (PSM 데이터셋)")
    print_comparison(no_revin_metrics, revin_metrics)
    print_train_curve(no_revin_losses, revin_losses)
    print()


if __name__ == '__main__':
    main()
