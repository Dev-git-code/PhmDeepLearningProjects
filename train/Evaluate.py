import sys
sys.path.append(".")
from dataset import utils

print("FD001 scores: \n")
utils.compute_metrics("train\model_result\RUL-STSF-h32-6-norm1-w30-batch1024-thresh125-FD001-neg0-1")
print("\nFD002 scores: \n")
utils.compute_metrics("train/model_result/RUL-STSF-h32-6-norm1-w30-batch1024-thresh125-FD002-neg0-1")
print("\nFD003 scores: \n")
utils.compute_metrics("train/model_result/RUL-STSF-h32-6-norm1-w30-batch1024-thresh125-FD003-neg0-1")
print("\nFD004 scores: \n")
utils.compute_metrics("train/model_result/RUL-STSF-h32-6-norm1-w30-batch1024-thresh125-FD004-neg0-1")
