import logging
import wandb
import time
import os
import json
import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict
from accelerate import Accelerator
from utils.utils import check_graph, Float32Encoder
from utils.tools import adjust_learning_rate, EarlyStopping
from utils.metrics import cal_metric, anomaly_metric, bf_search, calc_seq, get_best_f1, get_adjusted_composite_metrics, percentile_search, bf_search1, calc_seq1

_logger = logging.getLogger('train')

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def training_dl(
    model, trn_dataloader, val_dataloader, criterion, optimizer, accelerator: Accelerator, 
    savedir: str, epochs: int, eval_epochs: int, log_epochs: int, log_eval_iter: int, 
    use_wandb: bool, wandb_iter: int, ckp_metric: str, model_name: str, 
    early_stopping_metric: str, early_stopping_count: int,
    lradj: int, learning_rate: int, model_config: dict):
    
    # avereage meter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    # set mode
    model.train()
    optimizer.zero_grad()
    end_time = time.time()
    
    early_stopping = EarlyStopping(patience=early_stopping_count)
    
    # init best score and step
    best_score = np.inf
    wandb_iteration = 0
    
    _logger.info(f"\n 🔹 Training started")

    for epoch in range(epochs):
        epoch_time = time.time()
        for idx, item in enumerate(trn_dataloader):
            data_time_m.update(time.time() - end_time)

            """
            목적: 구성한 Dataloader를 바탕으로 모델의 입력을 구성
            조건
            - 구성한 Dataloader에 적합한 입력을 통하여 모델의 출력을 계산
            - 이상탐지 모델은 loss나 score를 계산하는 과정이 모델마다 상이할 수 있기에 모델 내부에서 계산
            - model은 LSTM_AE를 사용하고 있기 때문에, 코드 참고하여 작성
            - 모든 모델에서 모델만 변경할 경우 작동될 수 있도록 구현
            """
            # 데이터 추출 (BuildDataset의 dict 반환 형태에 맞게)
            inputs = item['sequence']
            targets = inputs  # Auto-encoder: 입력 = 타겟

            # 더미 타임스탬프 생성 (LSTM_AE forward signature에 맞게)
            batch_size_cur, seq_len_cur, _ = inputs.shape
            dummy_timestamps = torch.zeros(batch_size_cur, seq_len_cur).to(inputs.device)

            outputs, loss = model(inputs, dummy_timestamps, targets, criterion)

            loss = accelerator.gather(loss).mean()
            outputs, targets = accelerator.gather_for_metrics((outputs.contiguous(), targets.contiguous()))

            accelerator.backward(loss)

            # loss update
            optimizer.step()
            optimizer.zero_grad()

            losses_m.update(loss.item(), n=targets.size(0))

            # batch time
            batch_time_m.update(time.time() - end_time)
            wandb_iteration += 1

            if use_wandb and (wandb_iteration + 1) % wandb_iter:
                train_results = OrderedDict([
                    ('lr', optimizer.param_groups[0]['lr']),
                    ('train_loss', losses_m.avg)
                ])
                wandb.log(train_results, step=idx + 1)

        if (epoch + 1) % log_epochs == 0:
            _logger.info('EPOCH {:>3d}/{} | TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                         'LR: {lr:.3e} '
                         'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                         'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                         (epoch + 1), epochs,
                         (idx + 1),
                         len(trn_dataloader),
                         loss       = losses_m,
                         lr         = optimizer.param_groups[0]['lr'],
                         batch_time = batch_time_m,
                         rate       = inputs.size(0) / batch_time_m.val,
                         rate_avg   = inputs.size(0) / batch_time_m.avg,
                         data_time  = data_time_m))
            
                    
        if (epoch+1) % eval_epochs == 0:
            eval_metrics = test_dl(
                accelerator   = accelerator,
                model         = model, 
                dataloader    = val_dataloader, 
                criterion     = criterion,
                name          = 'VALID',
                log_interval  = log_eval_iter,
                savedir       = savedir,
                model_name    = model_name,
                model_config  = model_config,
                return_output = False,
                )

            model.train()
            
            # eval results
            eval_results = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
            
            # wandb
            if use_wandb:
                wandb.log(eval_results, step=idx+1)
                
            # check_point
            if best_score > eval_metrics[ckp_metric]:
                # save results
                state = {'best_epoch':epoch ,
                            'best_step':idx+1, 
                            f'best_{ckp_metric}':eval_metrics[ckp_metric]}
                
                print('Save best model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
                    to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))
                
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    state.update(eval_results)
                    json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), 
                                indent='\t', cls=Float32Encoder)

                # save model
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best {0} {1:6.6f} to {2:6.6f}'.format(ckp_metric.upper(), best_score, eval_metrics[ckp_metric]))
                    _logger.info("\n✅ Saved best model")
                best_score = eval_metrics[ckp_metric]
                
            early_stopping(eval_metrics[early_stopping_metric])
            if early_stopping.early_stop:
                _logger.info("⏳ Early stopping triggered")
                break
        
        adjust_learning_rate(optimizer, epoch + 1, lradj, learning_rate)

        end_time = time.time()

    # save latest model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

        print('Save latest model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
            to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))

        # save latest results
        state = {'best_epoch':epoch ,'best_step':idx+1, f'latest_{ckp_metric}':eval_metrics[ckp_metric]}
        state.update(eval_results)
        json.dump(state, open(os.path.join(savedir, f'latest_results.json'),'w'), indent='\t', cls=Float32Encoder)
    _logger.info("\n🎉 Training complete for all datasets")
    
def test_dl(model, dataloader, criterion, accelerator: Accelerator, log_interval: int, 
            savedir: str, model_config: dict, model_name: str, name: str = 'TEST', 
            return_output: bool = False, plot_result:bool = False) -> dict:
    _logger.info(f'\n[🔍 Start {name} Evaluation]')

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    total_label = []
    total_outputs = []
    total_score   = []
    total_targets = []
    total_timestamp = []
    history = dict()

    end_time = time.time()

    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            data_time_m.update(time.time() - end_time)

            """
            목적: 구성한 Dataloader를 바탕으로 모델의 입력을 구성
            조건
            - 구성한 Dataloader에 적합한 입력을 통하여 모델의 출력을 계산
            - model은 LSTM_AE를 사용하고 있기 때문에, 코드 참고하여 작성
            """
            # 데이터 추출 (BuildDataset의 dict 반환 형태에 맞게)
            inputs = item['sequence']
            targets = inputs  # Auto-encoder: 입력 = 타겟

            # 더미 타임스탬프 생성
            batch_size_cur, seq_len_cur, _ = inputs.shape
            dummy_timestamps = torch.zeros(batch_size_cur, seq_len_cur).to(inputs.device)

            # anomaly score 계산 포함하여 모델 호출
            outputs, loss, score = model(inputs, dummy_timestamps, targets, criterion, cal_score=True)

            loss = accelerator.gather(loss).mean()
            loss = torch.mean(loss)
            outputs, targets = accelerator.gather_for_metrics((outputs.contiguous(), targets.contiguous()))

            losses_m.update(loss.item(), n=inputs.size(0))
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            total_outputs.append(outputs)
            total_score.append(score)
            total_targets.append(targets)
            if 'timestamp' in item:
                total_timestamp.append(item['timestamp'].detach().cpu().numpy())

            if 'label' in item:
                label = item['label'].detach().cpu().numpy()
                total_label.append(label)

            batch_time_m.update(time.time() - end_time)

            if (idx+1) % log_interval == 0:
                _logger.info('{name} [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                (idx+1), 
                                len(dataloader),
                                name       = name, 
                                loss       = losses_m, 
                                batch_time = batch_time_m,
                                rate       = inputs.size(0) / batch_time_m.val,
                                rate_avg   = inputs.size(0) / batch_time_m.avg,
                                data_time  = data_time_m))

            end_time = time.time()

    
    """
    목적: 시계열 이상탐지 Task의 평가 지표 계산
    조건
    - 계산된 출력, 입력, label, score 등을 가지고, 시계열 이상탐지 metric 계산
    - 'metrics.py'의 cal_metric, bf_search, calc_seq 함수 참고하여 작성
    - 'VALID' 시에는 reconstruction loss만 도출
    """
    history = {'loss': losses_m.avg}

    if name == 'TEST' and len(total_label) > 0:
        all_labels = np.concatenate(total_label, axis=0).flatten()
        all_scores = np.concatenate(total_score, axis=0)

        # score 차원 축소: [N, features] → [N]
        if len(all_scores.shape) > 1:
            all_scores = np.mean(all_scores, axis=1)

        # 라벨과 스코어 길이 맞추기
        min_len = min(len(all_labels), len(all_scores))
        all_labels = all_labels[:min_len]
        all_scores = all_scores[:min_len]

        # Best-F1 threshold 탐색 (start=score 최솟값, end=최댓값, step_num=100)
        score_min = float(np.min(all_scores))
        score_max = float(np.max(all_scores))
        best_metrics, best_threshold = bf_search(
            score=all_scores, label=all_labels,
            start=score_min, end=score_max, step_num=100, verbose=False
        )
        # best_metrics: [f1, precision, recall, TP, TN, FP, FN, roc_auc, auprc, latency]
        best_f1, best_precision, best_recall = best_metrics[0], best_metrics[1], best_metrics[2]
        roc_auc = best_metrics[6] if len(best_metrics) > 6 else float('nan')
        auprc   = best_metrics[7] if len(best_metrics) > 7 else float('nan')

        history.update({
            'f1':        best_f1,
            'precision': best_precision,
            'recall':    best_recall,
            'roc_auc':   roc_auc,
            'auprc':     auprc,
            'threshold': best_threshold,
        })

        _logger.info(
            f'{name} Results - Loss: {history["loss"]:.4f}, '
            f'F1: {history["f1"]:.4f}, '
            f'Precision: {history["precision"]:.4f}, '
            f'Recall: {history["recall"]:.4f}'
        )
    else:
        _logger.info(f'{name} Results - Loss: {history["loss"]:.4f}')

    return history