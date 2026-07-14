import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import argparse
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
import base64
from openai import OpenAI
from swift import TransformersEngine, InferRequest, RequestConfig

Internvl3_5_SYSTEM_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()
def get_mime_type(img_path: str) -> str:
    """根据文件扩展名获取MIME类型"""
    ext = os.path.splitext(img_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    return mime_types.get(ext, 'image/jpeg')  # 默认jpeg


def encode_image_to_base64(image_path: str):
    """将图片编码为base64格式"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"警告：图片编码失败 {image_path}: {e}")
        return None


def load_raw_data(input_json_path: str, image_root_dir: str) -> List[Dict]:
    """
    加载原始数据，预处理图片绝对路径，校验数据格式
    返回格式：[{id, images, qa_pairs, ..., current_image_idx}]
    """
    # 校验路径合法性
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"输入json文件不存在: {input_json_path}")
    if not os.path.exists(image_root_dir):
        print(f"警告：图片根目录不存在: {image_root_dir}，可能导致图片加载失败")

    # 加载原始数据
    with open(input_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 整理数据
    case_list = []
    for item_idx, item in enumerate(raw_data):
        # 基础字段校验
        if "id" not in item or "images" not in item or "messages" not in item:
            print(f"警告：第{item_idx + 1}个样本缺少必填字段，已跳过")
            continue
        case_id = item["id"]
        raw_images = item["images"]
        messages = item["messages"]

        # 图片路径拼接
        if not isinstance(raw_images, list):
            print(f"警告：病例{case_id}的images不是列表格式，已跳过")
            continue
        absolute_images = []
        for img_path in raw_images:
            full_path = os.path.join(image_root_dir, img_path)
            absolute_images.append(full_path)
            if not os.path.exists(full_path):
                print(f"警告：病例{case_id}的图片不存在: {full_path}")

        # 整理问答对，校验对话格式
        if len(messages) % 2 != 0:
            print(f"警告：病例{case_id}的对话轮次不完整，user/assistant必须成对出现，已跳过该病例")
            continue
        qa_pairs = []
        for turn_idx in range(0, len(messages), 2):
            user_msg = messages[turn_idx]
            assistant_msg = messages[turn_idx + 1]
            if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
                print(f"警告：病例{case_id}第{turn_idx // 2 + 1}轮角色顺序错误，已跳过该轮")
                continue
            qa_pairs.append({
                "turn_id": turn_idx // 2 + 1,
                "user_content": user_msg["content"],
                "gold_standard": assistant_msg["content"],
                "model_prediction": None
            })

        # 添加到病例列表
        case_list.append({
            "id": case_id,
            "images": absolute_images,
            "qa_pairs": qa_pairs,
            "max_turn": len(qa_pairs),
            "current_turn": 1,
            "is_finished": False,
            "history_messages": [],
            "current_image_idx": 0  # 【新增】图片指针，记录当前用到第几张了
        })
    print(
        f"数据加载完成，共加载{len(case_list)}个独立病例，最大轮次为{max([case['max_turn'] for case in case_list], default=0)}轮")
    return case_list


def append_case_to_json(case: Dict, output_json_path: str):
    """将单个病例的当前状态直接追加写入JSON文件"""
    # 构建输出格式
    result_item = {
        "id": case["id"],
        "images": case["images"],
        "turns": []
    }

    # 按turn_id排序
    sorted_qa = sorted(case["qa_pairs"], key=lambda x: x["turn_id"])
    for qa in sorted_qa:
        result_item["turns"].append({
            "turn_id": qa["turn_id"],
            "user_content": qa["user_content"],
            "model_prediction": qa["model_prediction"],
            "gold_standard": qa["gold_standard"]
        })

    # 读取现有文件内容（如果存在）
    existing_results = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"警告：输出文件 {output_json_path} 格式异常，将覆盖重写")
            existing_results = []

    # 直接追加新结果（不做任何ID检查）
    existing_results.append(result_item)

    # 自动创建输出目录
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 写入文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=2)


def batch_inference_with_append_save(
        engine,
        case_list: List[Dict],
        request_config,
        batch_size: int,
        output_json_path: str,
        infer_backend: str,
        model_id_or_path: str
):
    """
    【核心逻辑重构】：支持 OpenAI 兼容格式和 Transformers 两种格式
    """

    # 1. 将总数据切分为多个大的 Batch
    total_batches = (len(case_list) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="总体Batch进度"):
        # 1.1 取出当前这个 Batch 的所有样本
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(case_list))
        batch_cases = case_list[start_idx:end_idx]

        # 1.2 找出这个 Batch 里最长的轮次数，决定我们要跑多少轮
        if not batch_cases:
            continue
        batch_max_turn = max([case["max_turn"] for case in batch_cases])

        # 1.3 针对这个 Batch，从第 1 轮到 max_turn 进行推理
        for current_turn in range(1, batch_max_turn + 1):
            # 筛选出当前 Batch 中还需要继续推理的样本
            active_cases = [case for case in batch_cases if
                            not case["is_finished"] and case["max_turn"] >= current_turn]

            if not active_cases:
                break  # 该 Batch 全员结束，提前退出

            # ============ 分支处理：OpenAI (vllm) 模式 ============
            if infer_backend == "vllm":
                for case in active_cases:
                    try:
                        current_qa = next(qa for qa in case["qa_pairs"] if qa["turn_id"] == current_turn)
                        user_text = current_qa["user_content"]

                        # ================== 核心修改开始 ==================
                        # 1. 统计当前轮有多少个 <image> 占位符
                        num_images_this_turn = user_text.count("<image>")

                        # 2. 根据指针切片取出对应的图片路径
                        start_idx = case["current_image_idx"]
                        end_idx = start_idx + num_images_this_turn
                        selected_image_paths = case["images"][start_idx:end_idx]

                        # 3. 更新指针，为下一轮做准备
                        case["current_image_idx"] = end_idx
                        # ================== 核心修改结束 ==================

                        messages = case["history_messages"].copy()
                        if current_turn == 1 and  args.thinking:
                            if "internvl" in args.model_id_or_path.low():
                                messages.insert(0, {"role": "system", "content": Internvl3_5_SYSTEM_PROMPT})
                        current_content = []

                        # 4. 将这一轮选中的图片编码并加入 content
                        for img_path in selected_image_paths:
                            base64_str = encode_image_to_base64(img_path)
                            if base64_str is None:
                                raise ValueError(f"图片编码失败: {img_path}")

                            mime_type = get_mime_type(img_path)
                            current_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_str}",
                                    # "detail": "low"#high
                                }
                            })

                        # 5. 加入文本 (可选：可以把 <image> 替换掉，也可以保留，通常不影响)
                        clean_text = user_text.replace("<image>", "").strip()
                        current_content.append({
                            "type": "text",
                            "text": clean_text
                        })

                        messages.append({
                            "role": "user",
                            "content": current_content
                        })

                        resp = engine.chat.completions.create(
                            model=model_id_or_path,
                            messages=messages,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            # temperature=0.7,
                            top_p=0.8,
                            presence_penalty=1.5,
                            extra_body={
                                "top_k": 20,
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                            # response_format={"type": "json_object"},
                            # timeout=API_TIMEOUT,
                        )

                        pred_content = resp.choices[0].message.content

                        # 更新状态
                        current_qa["model_prediction"] = pred_content
                        case["history_messages"].append({"role": "user", "content": current_content})
                        case["history_messages"].append({"role": "assistant", "content": pred_content})

                        # 检查是否结束并保存
                        if current_turn == case["max_turn"]:
                            case["is_finished"] = True
                            append_case_to_json(case, output_json_path)

                    except Exception as e:
                        print(f"Error processing case {case['id']}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            # ============ 分支处理：Transformers 模式 ============
            else:
                try:
                    # 构建推理请求
                    infer_requests = []
                    for case in active_cases:
                        current_qa = next(qa for qa in case["qa_pairs"] if qa["turn_id"] == current_turn)
                        user_text = current_qa["user_content"]

                        # Transformers 模式下也做同样的图片切片逻辑，保证逻辑一致
                        num_images_this_turn = user_text.count("<image>")
                        start_idx = case["current_image_idx"]
                        end_idx = start_idx + num_images_this_turn
                        selected_image_paths = case["images"][start_idx:end_idx]
                        case["current_image_idx"] = end_idx

                        # messages = case["history_messages"] + [{"role": "user", "content": user_text}]
                        messages = case["history_messages"].copy()
                        # 只在第一轮插入 system prompt
                        if current_turn == 1 and  args.thinking:
                            if "internvl" in args.model_id_or_path.low():
                                messages.insert(0, {"role": "system", "content": Internvl3_5_SYSTEM_PROMPT})
                        messages.append({"role": "user", "content": user_text})

                        infer_requests.append(InferRequest(
                            messages=messages,
                            images=selected_image_paths  # 这里也传切片后的图
                        ))

                    # 执行推理
                    batch_responses = engine.infer(
                        infer_requests=infer_requests,
                        request_config=request_config
                    )

                    # 解析结果并更新
                    for case, response in zip(active_cases, batch_responses):
                        current_qa = next(qa for qa in case["qa_pairs"] if qa["turn_id"] == current_turn)
                        pred_content = response.choices[0].message.content
                        current_qa["model_prediction"] = pred_content

                        # 更新历史 (Transformers 模式通常只存文本历史，视 SWIFT 要求而定)
                        # 这里为了简单，保持原逻辑，只存文本
                        user_text = current_qa["user_content"]
                        case["history_messages"].append({"role": "user", "content": user_text})
                        case["history_messages"].append({"role": "assistant", "content": pred_content})

                        # 检查是否结束
                        if current_turn == case["max_turn"]:
                            case["is_finished"] = True
                            append_case_to_json(case, output_json_path)
                except Exception as e:
                    print(e)
                    continue

    print(f"所有病例推理完成，结果已实时保存至: {output_json_path}")


def load_checkpoint(output_json_path: str) -> Set[Tuple[str, Tuple[str, ...]]]:
    """
    加载已有的输出文件作为 Checkpoint，返回已处理样本的标识集合
    并打印发现的重复样本
    """
    processed = set()
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                duplicate_count = 0

                for item in existing_results:
                    case_id = item["id"]
                    case_images = tuple(item["images"])
                    key = (case_id, case_images)

                    # 检查是否重复
                    if key in processed:
                        print(f"⚠️ 发现重复样本: ID = {case_id,case_images}")
                        duplicate_count += 1
                    else:
                        processed.add(key)

            print(f"✅ 加载 Checkpoint 成功，发现已处理 {len(processed)} 个样本")
            if duplicate_count > 0:
                print(f"⚠️ 其中包含 {duplicate_count} 个重复记录")

        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"⚠️ 警告：加载 Checkpoint 失败，将从头开始处理。错误信息: {e}")
    return processed




if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    parser = argparse.ArgumentParser(description="多模态多轮批量推理脚本 (兼容 OpenAI & Transformers)")

    # 推理后端配置
    parser.add_argument("--infer_backend", type=str, default="vllm", choices=["vllm", "transformers"],
                        help="推理后端，vllm使用OpenAI格式请求，transformers使用SWIFT")

    # 模型配置
    parser.add_argument("--model_id_or_path", type=str, default='/home/user02/SCY/SonoVLM_V2/checkpoints/ablation/ablation_data/data_scaling/checkpoint-1170-merged',
                        help="模型id或本地路径")
    parser.add_argument("--model_type", type=str, default='qwen3_vl_moe', choices=['internvl', 'qwen2_5_vl',
                                                                                 'qwen3_vl', 'gemma3_vision','qwen3_vl',
                                                                                 'minicpmv4_5', 'qwen3_vl_moe','qwen3_5'
                                                                                 'kimi_vl', 'llava1_6_vicuna_hf'],
                        help="SWIFT预定义模型类型")
    parser.add_argument("--template", type=str, default='qwen3_vl', choices=['internvl_hf', 'qwen2_5_vl',
                                                                               'qwen3_vl', 'gemma3_vision','qwen3_5'
                                                                               'minicpmv4_5','qwen3_vl_moe',
                                                                               'kimi_vl', 'llava1_6_vicuna_hf', ],
                        help="对话模板类型")

    # OpenAI 特有配置
    parser.add_argument("--base_url", type=str, default="http://10.116.39.70:8001/v1",
                        help="OpenAI API Base URL")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="OpenAI API Key")

    # 多卡分布式配置
    parser.add_argument("--device_map", type=str, default="cuda:2",
                        help="transformers模式设备映射，可选：auto、balanced、sequential、cuda:0等")

    # 显存与性能配置
    parser.add_argument("--batch_size", type=int, default=16,
                        help="单批次推理的样本数量")

    # 数据路径配置
    parser.add_argument("--image_root_dir", type=str, default='/home/user02/SCY/thyroid_benchmark_desensitization',
                        help="图片根目录")
    parser.add_argument("--input_json", type=str,
                        default='/home/user02/SCY/thyroid_benchmark/code/dataset_splits_final/test/qa_test_dataset/english/OCR_Multiple-Choice_test.json',
                        help="输入json文件路径")
    parser.add_argument("--output_json", type=str,#Multiple-Choice，Single-Choice，True-False,caption,finding,report
                        default='/home/user02/SCY/thyroid_benchmark/code/test_result/echovlm/EchoVLM_OCR_Multiple-Choice.json',
                        help="输出结果json路径")

    # 推理生成配置
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--thinking", type=bool, default=False,
                        help="采样温度")

    args = parser.parse_args()
    print(args)

    # 1. 加载并预处理数据
    all_cases = load_raw_data(args.input_json, args.image_root_dir)
    if not all_cases:
        print("没有有效病例数据，程序退出")
        exit()

    # 2. 加载 Checkpoint
    processed_set = load_checkpoint(args.output_json)
    case_list_to_process = []
    for case in all_cases:
        case_key = (case["id"], tuple(case["images"]))
        if case_key not in processed_set:
            case_list_to_process.append(case)

    skipped_count = len(all_cases) - len(case_list_to_process)
    print(
        f"🔍 Checkpoint 校验完成，原始数据: {len(all_cases)} 条，将跳过: {skipped_count} 条，剩余: {len(case_list_to_process)} 条\n")

    if not case_list_to_process:
        print("🎉 所有数据均已处理完成，程序退出。")
        exit()

    if args.infer_backend == "vllm":
        print(f"正在初始化 OpenAI Client，连接至: {args.base_url}...")
        engine = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
        )
        request_config = None
    else:
        print(f"正在初始化 Transformers 推理引擎，模板类型：{args.template}...")
        engine = TransformersEngine(
            model=args.model_id_or_path,
            model_type=args.model_type,
            template_type=args.template,
            device_map=args.device_map,
            max_batch_size=args.batch_size,
            attn_impl='flash_attn'
        )
        request_config = RequestConfig(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stream=False
        )

    print(f"推理引擎初始化完成")

    # 4. 执行推理逻辑
    batch_inference_with_append_save(
        engine=engine,
        case_list=case_list_to_process,
        request_config=request_config,
        batch_size=args.batch_size,
        output_json_path=args.output_json,
        infer_backend=args.infer_backend,
        model_id_or_path=args.model_id_or_path
    )
#CUDA_VISIBLE_DEVICES=4,6 setsid vllm serve /home/user02/SCY/Model/Hulu-30A3 --tensor-parallel-size 2 --gpu-memory-utilization 0.9  --port 6002 --limit-mm-per-prompt '{"image": 100,"video": 1}' --max-num-seqs 128 --max-model-len 32768 > /home/user02/SCY/thyroid_benchmark/code/swift/serve.log 2>&1