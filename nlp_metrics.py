import json
import os
import jieba
import torch
import argparse
import nltk
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
# 最终选定的导入
from torchmetrics.text import SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore
from bert_score import BERTScorer
import warnings
import re

warnings.filterwarnings("ignore")

# 自动下载nltk资源（解决ROUGE和英文分词依赖）
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    print("正在下载nltk punkt资源...")
    nltk.download("punkt_tab", quiet=True)
    print("nltk punkt资源下载完成！\n")

# 全局初始化jieba
jieba.initialize()


# jieba.load_userdict("medical_terms_dict.txt")  # 有医学词典可取消注释

# -------------------------- 语言检测工具 --------------------------
def detect_language(text: str) -> str:
    """
    简单高效的中英文语言检测
    返回: "zh" 或 "en"
    """
    # 统计中文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 统计英文字符数量
    english_chars = len(re.findall(r'[a-zA-Z]', text))

    # 如果中文字符占比超过30%，判定为中文
    total_chars = chinese_chars + english_chars
    if total_chars == 0:
        return "zh"  # 默认中文

    if chinese_chars / total_chars > 0.3:
        return "zh"
    else:
        return "en"


def get_tokenizer(language: str):
    """根据语言返回对应的分词器"""
    if language == "zh":
        return jieba.lcut
    elif language == "en":
        return nltk.word_tokenize
    else:
        raise ValueError(f"不支持的语言: {language}")


# -------------------------- 数据集类 --------------------------
class NLGDataset(Dataset):
    def __init__(self, json_path: str, language: str = "auto"):
        self.samples = []
        self.empty_count = 0
        self.filename = os.path.basename(json_path)
        self.language = language
        self.detected_language = None  # 自动检测后的实际语言

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            for turn in item.get("turns", []):
                pred = turn.get("model_prediction", "").strip()
                ref = turn.get("gold_standard", "").strip()

                if pred and ref:
                    # 自动检测语言（只检测第一个有效样本）
                    if self.language == "auto" and self.detected_language is None:
                        self.detected_language = detect_language(ref)
                        print(f"🔍 自动检测到语言: {'中文' if self.detected_language == 'zh' else '英文'}")

                    self.samples.append({
                        "prediction": pred,
                        "reference": ref
                    })
                else:
                    self.empty_count += 1
                    print(f"空样本id:{item['id']}")

        # 如果是自动模式但没有检测到语言（所有样本为空），默认中文
        if self.language == "auto" and self.detected_language is None:
            self.detected_language = "zh"

        # 打印数据集统计信息
        self._print_stats()

    def _print_stats(self):
        """打印数据集统计信息"""
        print(f"📂 加载文件: {self.filename}")
        print(f"   总样本数: {len(self.samples) + self.empty_count}")
        print(f"   有效样本: {len(self.samples)}")
        if self.empty_count > 0:
            print(f"   空样本数: {self.empty_count} (已跳过)")
        if self.language != "auto":
            print(f"   指定语言: {'中文' if self.language == 'zh' else '英文'}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pred = sample["prediction"]
        ref = sample["reference"]

        # 关键修改：处理prediction中的think标签
        # 先判断
        if "</think>" in pred:
            pred = pred.split("</think>", 1)[-1]
        # 再判断 ◁/think▷
        elif "◁/think▷" in pred:
            pred = pred.split("◁/think▷", 1)[-1]

        # 执行去除空字符串、文本开头的换行和开头的空格操作
        pred = pred.lstrip("\n ")  # 去除开头的换行和空格

        # 返回处理后的样本
        return {
            "prediction": pred,
            "reference": ref
        }

    def get_actual_language(self):
        """获取实际使用的语言"""
        if self.language == "auto":
            return self.detected_language
        return self.language


# -------------------------- 评估器类 --------------------------
class NLGEvaluator:
    def __init__(
            self,
            batch_size: int = 100,
            device: str = "cuda:2",
            bert_model: str = "/home/user02/SCY/Model/ModernBERT-large",
            language: str = "auto"
    ):
        self.batch_size = batch_size
        self.device = device
        self.bert_model = bert_model
        self.language = language

        # 初始化BERTScore（预加载模型权重，仅加载一次，中英文通用）
        print("正在预加载BERTScore模型...")
        self.bert_scorer = BERTScorer(
            model_type=self.bert_model,
            device=self.device,
            batch_size=self.batch_size,
            num_layers=28
        )
        print("✅ BERTScore模型加载完成！")

        # 其他指标将在评估时根据语言动态初始化
        self.sacre_bleu = None
        self.rouge = None

        print(f"✅ 评估器初始化完成！使用设备: {self.device}")
        print(f"   语言模式: {'自动检测' if language == 'auto' else ('中文' if language == 'zh' else '英文')}")
        print("已加载指标: SacreBLEU-1, ROUGE-1/2/L, BERTScore-F1\n")

    def _init_metrics_for_language(self, language: str):
        """根据语言初始化对应的评估指标"""
        print(f"🔧 为{'中文' if language == 'zh' else '英文'}初始化评估指标...")

        # TorchMetrics: SacreBLEU-1
        if language == "zh":
            self.sacre_bleu = SacreBLEUScore(n_gram=1, tokenize="zh", lowercase=True).to(self.device)
        else:  # en
            self.sacre_bleu = SacreBLEUScore(n_gram=1, tokenize="13a", lowercase=True).to(self.device)

        # TorchMetrics: ROUGE-1/2/L
        tokenizer = get_tokenizer(language)
        self.rouge = ROUGEScore(
            tokenizer=tokenizer,
            rouge_keys=("rouge1", "rouge2", "rougeL")
        ).to(self.device)

        print("✅ 评估指标初始化完成！")

    def _evaluate_single_file(self, json_path: str):
        """内部方法：评估单个JSON文件"""
        filename = os.path.basename(json_path)

        # 加载数据集
        try:
            dataset = NLGDataset(json_path, self.language)
        except Exception as e:
            print(f"❌ 加载文件 {filename} 失败: {e}\n")
            return

        if len(dataset) == 0:
            print(f"⚠️ {filename} 没有有效样本，跳过评估\n")
            return

        # 获取实际使用的语言
        actual_language = dataset.get_actual_language()

        # 根据语言初始化指标
        self._init_metrics_for_language(actual_language)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # 重置指标
        self.sacre_bleu.reset()
        self.rouge.reset()
        all_bert_scores = []

        # 批量累积计算
        with torch.autocast(device_type=self.device.split(':')[0]):  # 兼容cuda和cuda:x格式
            for batch in tqdm(dataloader, desc="评估进度"):
                # 通用兼容处理：同时支持列表和张量输入
                preds = batch["prediction"]
                refs = batch["reference"]
                # TorchMetrics: SacreBLEU + ROUGE
                self.sacre_bleu.update(preds, [[r] for r in refs])
                self.rouge.update(preds, refs)

                # 官方库: BERTScore（中英文通用）
                _, _, f1 = self.bert_scorer.score(preds, refs, batch_size=self.batch_size)
                all_bert_scores.extend(f1.cpu().numpy())

        # 计算最终结果
        final_bleu = self.sacre_bleu.compute().item()
        final_rouge = self.rouge.compute()
        final_bert = np.mean(all_bert_scores)

        # 打印结果
        final_bleu_pct = final_bleu * 100
        final_rouge1_pct = final_rouge['rouge1_fmeasure'].item() * 100
        final_rouge2_pct = final_rouge['rouge2_fmeasure'].item() * 100
        final_rougeL_pct = final_rouge['rougeL_fmeasure'].item() * 100
        final_bert_pct = final_bert * 100

        # Excel友好输出：横排制表符分隔，直接复制粘贴
        print("\n📊 评估结果（可直接复制到Excel）")
        print("-" * 120)
        print("模型名称\t语言\tSacreBLEU-1\tROUGE-1\tROUGE-2\tROUGE-L\tBERTScore-F1\t评估样本数")
        print(
            f"{filename}\t{'中文' if actual_language == 'zh' else '英文'}\t{final_bleu_pct:.2f}\t{final_rouge1_pct:.2f}\t{final_rouge2_pct:.2f}\t{final_rougeL_pct:.2f}\t{final_bert_pct:.2f}\t{len(dataset)}")
        print("-" * 120 + "\n")

    def evaluate(self, input_path: str):
        """统一评估入口：自动处理单个文件或目录"""
        if os.path.isfile(input_path) and input_path.endswith(".json"):
            self._evaluate_single_file(input_path)

        elif os.path.isdir(input_path):
            json_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".json")]
            if not json_files:
                print("❌ 输入目录中没有找到JSON文件")
                return

            print(f"找到 {len(json_files)} 个文件待处理\n")
            for json_file in json_files:
                self._evaluate_single_file(json_file)

        else:
            print("❌ 输入路径无效，请提供有效的JSON文件或目录")


# -------------------------- 命令行入口 --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中英文NLG生成质量评估工具")

    # 默认参数配置
    parser.add_argument("--input_path", type=str,
                        default="/home/user02/SCY/thyroid_benchmark/code/test_result/ablation/stf_v6_700/Qwen3.5-9B_report_generation.json",
                        help="输入路径：单个JSON文件或包含JSON文件的目录")#caption,finding,report
    parser.add_argument("--batch_size", type=int, default=100,
                        help="批量大小 (默认: 100)")
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu",
                        help="计算设备 (默认: cuda:2)")
    parser.add_argument("--bert_model", type=str, default="/home/user02/SCY/Model/ModernBERT-large",
                        help="BERT模型路径 (默认: ModernBERT-large)")
    parser.add_argument("--language", type=str, default="auto", choices=["auto", "zh", "en"],
                        help="语言模式: auto(自动检测), zh(中文), en(英文) (默认: auto)")

    args = parser.parse_args()

    # 创建评估器实例并执行评估
    evaluator = NLGEvaluator(
        batch_size=args.batch_size,
        device=args.device,
        bert_model=args.bert_model,
        language=args.language
    )

    evaluator.evaluate(args.input_path)