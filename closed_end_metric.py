import json
import re
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import ast


def extract_options(text):
    """从文本中提取所有大写英文字母选项(A,B,C,D...)，排除单词内部和冒号后的字母"""
    # 正则说明：
    # (?<![A-Za-z:]) 前置排除：前面不能是英文字母或冒号（解决冒号后字母误提取）
    # [A-D]+         匹配一个或多个连续的大写A-D选项
    # (?![A-Za-z])   后置排除：后面不能是英文字母（解决单词内部字母误提取）
    matches = re.findall(r'(?<![A-Za-z:])[A-D]+(?![A-Za-z])', text)
    # 将连续的选项字符串（如"ABC"）拆分为单个字母，并去重排序
    options = []
    for match in matches:
        options.extend(list(match))
    return sorted(list(set(options)))


def extract_boolean(text):
    """
    【增强版】从文本中提取判断题结果
    兼容：中文【正确/错误】、英文【true/false】、【yes/no】，全程不区分大小写
    统一返回标准格式："正确" / "错误" / None
    """
    if not text:
        return None
    # 统一转为小写，实现大小写不敏感匹配，兼容TRUE/False/YES/No等各种写法
    text_lower = text.lower()

    # 匹配肯定类结果（优先级：中文>英文）
    if "正确" in text or "true" in text_lower or "yes" in text_lower or "对" in text_lower or "是的" in text_lower:
        return "正确"
    # 匹配否定类结果
    elif "错误" in text or "false" in text_lower or "no" in text_lower or "错" in text_lower or "不是" in text_lower or "没有" in text_lower:
        return "错误"
    # 无有效匹配结果
    else:
        return None


def clean_model_prediction(pred_str):
    if not pred_str:
        return ""

    # 1. 预处理：去除思考标签 (保留你原有的逻辑)
    if "</think>" in pred_str:
        pred_str = pred_str.split("</think>", 1)[-1]
    # 注意：你原代码里有两个 </think> 判断，这里合并逻辑或保留均可，建议保留以防万一
    elif "◁/think▷" in pred_str:
        pred_str = pred_str.split("◁/think▷", 1)[-1]
    elif "<unused95>" in pred_str:
        pred_str = pred_str.split("<unused95>", 1)[-1]

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', pred_str)
    if boxed_match:
        content = boxed_match.group(1).strip()
        # 如果内容是 "\text{C}" 或其他格式，提取第一个字母
        letter_match = re.search(r'[A-Za-z]', content)
        if letter_match:
            return letter_match.group(0).upper()
        return content

    # 2. 尝试提取 JSON 内容 (分优先级)

    # 优先级 A: 提取 ```json ... ``` 代码块中的内容
    # 这里的 group(1) 会拿到代码块里面的文字
    json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", pred_str)

    if json_block_match:
        cleaned_json_str = json_block_match.group(1)
    else:
        # 优先级 B: 兜底策略 - 直接在长文本中搜索包含 "answer" 的 JSON 对象 {...}
        # 解释：\{ 匹配左大括号，[^{}]* 匹配中间非大括号字符，"answer" 匹配关键字，\} 匹配右大括号
        # 这个正则会直接找到 {"answer": "C"} 这一整段
        direct_json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', pred_str)
        # direct_json_match = re.search(r'\{"answer"\s*:\s*"([^"]+)"\}', pred_str)

        if direct_json_match:
            cleaned_json_str = direct_json_match.group(0)  # group(0) 是整个匹配到的字符串
        else:
            # 优先级 C: 最后的兜底 - 尝试清理首尾的 markdown 标记
            cleaned_json_str = re.sub(r'^```(json)?\s*', '', pred_str, flags=re.IGNORECASE)
            cleaned_json_str = re.sub(r'\s*```$', '', cleaned_json_str)

    # 3. 解析 JSON 并提取 answer
    try:
        # 尝试将字符串转为字典
        # 此时 cleaned_json_str 应该是 '{"answer": "C"}' 这种格式
        data = json.loads(cleaned_json_str)

        # 如果解析成功，直接返回 answer 字段
        if isinstance(data, dict):
            return data.get("answer", "")
        elif isinstance(data, list) and len(data) > 0:
            # 有时候模型会输出 [{"answer": "C"}]，做个列表兼容
            return data[0].get("answer", "")

    except json.JSONDecodeError:
        # 如果 JSON 解析失败（比如格式不标准，或者 cleaned_json_str 里还有多余文字）
        # 我们再用正则硬抠一下 "answer" 的值
        # 正则：匹配 "answer" : "值"
        val_match = re.search(r'["\']answer["\']\s*:\s*["\']([^"\']+)["\']', cleaned_json_str)
        if val_match:
            return val_match.group(1)

    # 如果所有方法都失败，返回清洗后的字符串作为最终兜底
    return cleaned_json_str
# def clean_model_prediction(pred_str):
#
#     if not pred_str:
#         return ""
#     if "</think>" in pred_str:
#         pred_str = pred_str.split("</think>", 1)[-1]
#     # 再判断 ◁/think▷
#     elif "◁/think▷" in pred_str:
#         pred_str = pred_str.split("◁/think▷", 1)[-1]
#     elif "<unused95>" in pred_str:
#         pred_str = pred_str.split("<unused95>", 1)[-1]
#     # 移除开头的代码块标记 (支持 ```json 和 ```)
#     json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", pred_str)
#     if json_match:
#         cleaned = json_match.group(1)
#     else:
#         cleaned = re.sub(r'^```(json)?\s*', '', pred_str, flags=re.IGNORECASE)
#         # 移除结尾的代码块标记
#         cleaned = re.sub(r'\s*```$', '', cleaned)
#     # cleaned = re.sub(r'\s+', '', cleaned)
#     # 移除可能的单引号和反引号
#     # cleaned = cleaned.replace("'", '"').replace('`', '')
#
#     return cleaned


def parse_model_prediction(pred_str):
    """
    增强版解析函数：
    1. 先清理干扰字符，再尝试解析JSON
    2. 同时支持英文key "answer" 和中文key "答案"
    3. 支持三种格式：
       - 字典格式：{"answer": "A"} / {"答案": ["A", "B"]} / {"答案": "ABCXX"}
       - 列表格式：["A", "B"] / ["A", "B", "C"]
       - 纯文本格式：兜底方案
    """
    # 第一步：清理干扰字符
    cleaned_pred = clean_model_prediction(pred_str)
    try:
        if isinstance(cleaned_pred, (dict, list)):
            pred_json = cleaned_pred
        elif isinstance(cleaned_pred, str):
            print(f"直接返回字符串：{cleaned_pred}")
            return str(cleaned_pred)
            # pred_json = json.loads(cleaned_pred)

        # 情况1: 如果解析结果是字典（同时支持"answer"和"答案"两个key）
        if isinstance(pred_json, dict):
            # 优先取英文key "answer"，不存在则取中文key "答案"
            answer = pred_json.get("answer") or pred_json.get("答案", "")

            # answer可能是字符串 "A" / "ABC" / "正确" 或列表 ["A", "B"]
            if isinstance(answer, list):
                # 提取列表中的选项并排序去重
                options = []
                for item in answer:
                    if isinstance(item, str):
                        options.extend(extract_options(item))
                return ",".join(sorted(set(options))) if options else ""
            else:
                # answer是字符串，直接返回（后续会根据题型处理）
                return str(answer).strip()

        # 情况2: 如果解析结果是列表 ["A", "B"]
        elif isinstance(pred_json, list):
            options = []
            for item in pred_json:
                if isinstance(item, str):
                    options.extend(extract_options(item))
            return ",".join(sorted(set(options))) if options else ""

        # 情况3: 其他类型，转为字符串处理
        else:
            print(f"直接返回字符串：{pred_json}")
            return str(pred_json)

    except Exception :
        try:
            if isinstance(cleaned_pred, (dict, list)):
                pred_json = cleaned_pred
            else:
                pred_json = ast.literal_eval(cleaned_pred)
            answer=pred_json.get("answer") or pred_json.get("答案", "")
            return str(answer).strip()
        except Exception :
                # 如果JSON解析失败，尝试直接从原始文本中提取答案(兜底方案)
            if args.task_types !='boolean':
                options = extract_options(cleaned_pred)
                print(f"JSON解析失败，原始结果{cleaned_pred}----->二次解析结果:{options}")
                if options:
                    return ",".join(options)
            else:
                bool_result = extract_boolean(cleaned_pred)
                print(f"JSON解析失败，原始结果{cleaned_pred}----->二次解析结果:{bool_result}")
                if bool_result:
                    return bool_result
        return ""


def get_wrong_answer(gold_standard, task_type):
    """
    根据正确答案和题型，生成一个明确错误的答案
    用于将解析失败的案例强制计为错误

    Args:
        gold_standard: 标准答案字符串
        task_type: 题型 ("single", "boolean", "multi")

    Returns:
        与正确答案不同的错误答案
    """
    if task_type == "single":
        # 单选题：从ABCD中选一个与正确答案不同的选项
        true_options = extract_options(gold_standard)
        true_option = true_options[0] if true_options else "A"
        # 备选错误答案池
        all_options = ['A', 'B', 'C', 'D']
        for opt in all_options:
            if opt != true_option:
                return opt  # 返回第一个不同的选项即可
        return "A"  # 兜底

    elif task_type == "boolean":
        # 判断题：返回与正确答案相反的结果
        true_bool = extract_boolean(gold_standard)
        if true_bool == "正确":
            return "错误"
        else:
            # 如果正确答案是"错误"或无法解析，则返回"正确"作为错误预测
            return "正确"

    elif task_type == "multi":
        # 多选题：返回与正确答案不同的选项组合
        true_options = extract_options(gold_standard)
        all_options = ['A', 'B', 'C', 'D']
        # 选择第一个不在正确答案中的选项作为错误预测
        wrong_options = [opt for opt in all_options if opt not in true_options]
        if wrong_options:
            return [wrong_options[0]]
        # 如果正确答案包含所有选项，则返回空列表（不选任何选项）
        return []

    return ""


def evaluate_json_file(file_path, task_type):
    """
    解析JSON文件并计算评估指标
    task_type: "single", "boolean", "multi"

    核心逻辑：
    - 解析成功的样本：按实际预测结果计算
    - 解析失败的样本：强制计为错误（赋予一个与正确答案不同的预测值）
    """
    # 初始化数据存储
    y_true = []
    y_pred = []
    failed_cases = []  # 存储所有失败案例详情

    total_samples = 0
    failed_samples = 0

    # 读取并解析JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历每条数据
    for item_idx, item in enumerate(data):
        # 获取数据集中的 id 字段，如果没有则用索引
        data_id = item.get("id", f"index_{item_idx}")
        turns = item.get("turns", [])

        # 遍历每个对话轮次
        for turn_idx, turn in enumerate(turns):
            total_samples += 1
            sample_id = f"ID:{data_id}"
            model_pred_str = turn.get("model_prediction", "")
            gold_standard = turn.get("gold_standard", "")

            # ----------------------------------------------------------------
            # 解析模型预测结果
            # ----------------------------------------------------------------
            model_answer = parse_model_prediction(model_pred_str)

            # ----------------------------------------------------------------
            # 场景1：模型预测解析为空 → 强制计为错误
            # ----------------------------------------------------------------
            if not model_answer:
                failed_samples += 1
                fail_reason = "解析模型预测失败，未提取到有效答案"

                # 生成一个错误的预测值，确保该样本被计为错误
                wrong_pred = get_wrong_answer(gold_standard, task_type)

                failed_cases.append({
                    "sample_id": sample_id,
                    "model_pred_raw": model_pred_str,
                    "gold_standard": gold_standard,
                    "fail_reason": fail_reason,
                    "forced_pred": wrong_pred  # 记录被强制赋予的错误预测
                })

                # 将强制错误答案加入评估列表
                _add_to_eval(
                    task_type, gold_standard, wrong_pred,
                    y_true, y_pred, is_forced=True
                )
                continue

            # ----------------------------------------------------------------
            # 场景2：根据题型解析预测结果
            # ----------------------------------------------------------------
            valid = True
            fail_reason = ""

            if task_type == "single":
                pred_option = extract_options(model_answer)
                true_option = extract_options(gold_standard)
                if pred_option and true_option:
                    y_pred.append(pred_option[0])
                    y_true.append(true_option[0])
                else:
                    valid = False
                    fail_reason = "单选题未提取到有效选项"

            elif task_type == "boolean":
                pred_bool = extract_boolean(model_answer)
                true_bool = extract_boolean(gold_standard)
                if pred_bool and true_bool:
                    y_pred.append(pred_bool)
                    y_true.append(true_bool)
                else:
                    valid = False
                    fail_reason = "判断题未提取到【正确/错误/true/false/yes/no】有效结果"

            elif task_type == "multi":
                pred_options = extract_options(model_answer)
                true_options = extract_options(gold_standard)
                y_pred.append(pred_options)
                y_true.append(true_options)

            # ----------------------------------------------------------------
            # 场景3：题型解析失败 → 强制计为错误
            # ----------------------------------------------------------------
            if not valid:
                failed_samples += 1
                wrong_pred = get_wrong_answer(gold_standard, task_type)

                failed_cases.append({
                    "sample_id": sample_id,
                    "model_pred_raw": model_pred_str,
                    "gold_standard": gold_standard,
                    "fail_reason": fail_reason,
                    "forced_pred": wrong_pred
                })

                # 将强制错误答案加入评估列表
                _add_to_eval(
                    task_type, gold_standard, wrong_pred,
                    y_true, y_pred, is_forced=True
                )

    # ----------------------------------------------------------------
    # 计算评估指标
    # ----------------------------------------------------------------
    results = {
        "total_samples": total_samples,
        "failed_samples": failed_samples,
        "task_type": task_type,
        "metrics": {},
        "failed_cases": failed_cases
    }

    if y_true:
        results["metrics"]["num_samples"] = len(y_true)

        if task_type == "single":
            all_labels = ['A', 'B', 'C', 'D']
            results["metrics"]["accuracy"] = accuracy_score(y_true, y_pred)
            results["metrics"]["precision"] = precision_score(
                y_true, y_pred,
                labels=all_labels,
                average='macro',
                zero_division=0
            )
            results["metrics"]["recall"] = recall_score(
                y_true, y_pred,
                labels=all_labels,
                average='macro',
                zero_division=0
            )
            results["metrics"]["f1"] = f1_score(
                y_true, y_pred,
                labels=all_labels,
                average='macro',
                zero_division=0
            )

        elif task_type == "boolean":
            bool_map = {"正确": 1, "错误": 0}
            y_true_num = [bool_map[x] for x in y_true]
            y_pred_num = [bool_map[x] for x in y_pred]

            results["metrics"]["accuracy"] = accuracy_score(y_true_num, y_pred_num)
            results["metrics"]["precision"] = precision_score(
                y_true_num, y_pred_num, zero_division=0
            )
            results["metrics"]["recall"] = recall_score(
                y_true_num, y_pred_num, zero_division=0
            )
            results["metrics"]["f1"] = f1_score(
                y_true_num, y_pred_num, zero_division=0
            )

        elif task_type == "multi":
            mlb = MultiLabelBinarizer(classes=['A', 'B', 'C', 'D'])
            y_true_bin = mlb.fit_transform(y_true)
            y_pred_bin = mlb.transform(y_pred)
            results["metrics"]["accuracy"] = accuracy_score(y_true_bin, y_pred_bin)
            results["metrics"]["precision"] = precision_score(
                y_true_bin, y_pred_bin, average='macro', zero_division=0
            )
            results["metrics"]["recall"] = recall_score(
                y_true_bin, y_pred_bin, average='macro', zero_division=0
            )
            results["metrics"]["f1"] = f1_score(
                y_true_bin, y_pred_bin, average='macro', zero_division=0
            )

    return results


def _add_to_eval(task_type, gold_standard, wrong_pred, y_true, y_pred, is_forced=False):
    """
    辅助函数：将强制错误答案添加到评估列表
    确保失败案例以错误预测的形式参与指标计算

    Args:
        task_type:      题型
        gold_standard:  标准答案原始字符串
        wrong_pred:     强制赋予的错误预测值
        y_true:         真实标签列表（会被原地修改）
        y_pred:         预测标签列表（会被原地修改）
        is_forced:      是否为强制错误（保留扩展用）
    """
    if task_type == "single":
        true_options = extract_options(gold_standard)
        if true_options:
            y_true.append(true_options[0])
            # wrong_pred 已经是一个与正确答案不同的字母
            y_pred.append(wrong_pred if isinstance(wrong_pred, str) else wrong_pred[0])

    elif task_type == "boolean":
        true_bool = extract_boolean(gold_standard)
        if true_bool:
            y_true.append(true_bool)
            y_pred.append(wrong_pred)  # wrong_pred 是"正确"或"错误"

    elif task_type == "multi":
        true_options = extract_options(gold_standard)
        y_true.append(true_options)
        # wrong_pred 是一个列表
        y_pred.append(wrong_pred if isinstance(wrong_pred, list) else [wrong_pred])


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/user02/SCY/thyroid_benchmark/code/test_result/echovlm/EchoVLM_OCR_Multiple-Choice.json",
        help="JSON文件路径"#,Multiple-Choice，Single-Choice,True-False
    )
    parser.add_argument(
        "--task_types",default="multi",choices=["single", "boolean", "multi"],
        help="指定要评估的题型: single(单选题), boolean(判断题), multi(不定项选择题)"
    )

    args = parser.parse_args()

    # 运行评估
    evaluation_results = evaluate_json_file(args.file_path, args.task_types)

    # ----------------------------------------------------------------
    # 打印基本信息
    # ----------------------------------------------------------------
    print("=" * 100)
    print(
        f"Total Samples: {evaluation_results['total_samples']}  |  "
        f"Failed Samples: {evaluation_results['failed_samples']}  |  "
        f"（失败样本已强制计为错误参与指标计算）"
    )
    print("=" * 100)
    print("\n【可直接复制到Excel的结果】")
    print("-" * 100)

    # 定义表头和指标顺序
    metrics_order = ["num_samples", "accuracy", "precision", "recall", "f1"]
    task_names = {
        "single": "Single Choice",
        "boolean": "True/False",
        "multi": "Multiple Choice"
    }

    metrics = evaluation_results["metrics"]

    if metrics:
        # 打印表头（制表符分隔，方便粘贴到Excel）
        header = ["Task Type"] + metrics_order
        print("\t".join(header))

        # 打印数据行
        row_data = [task_names[args.task_types]]
        for metric in metrics_order:
            value = metrics.get(metric, 0)
            if metric == "num_samples":
                row_data.append(str(value))
            else:
                # 百分比形式，保留两位小数，不带%符号
                row_data.append(f"{value * 100:.2f}")

        print("\t".join(row_data))
    else:
        print("No valid sample data")

    # ----------------------------------------------------------------
    # 打印失败案例详情
    # ----------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"【失败案例详情（共{len(evaluation_results['failed_cases'])}条，均已强制计为错误）】")
    print("=" * 100)
    print(args.file_path)
    failed_cases = evaluation_results["failed_cases"]
    if failed_cases:
        for idx, case in enumerate(failed_cases, 1):
            print(f"\n第{idx}个失败案例：")
            print(f"  样本标识  ：{case['sample_id']}")
            print(f"  失败原因  ：{case['fail_reason']}")
            print(f"  标准答案  ：{case['gold_standard']}")
            print(f"  模型原始预测：{case['model_pred_raw']}")
            print(f"  强制错误预测：{case['forced_pred']}")
            print("-" * 80)
    else:
        print("✅ 无失败案例！")

    print("=" * 100)
#ksiehva#@821