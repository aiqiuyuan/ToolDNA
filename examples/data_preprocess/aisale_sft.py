import argparse
import os
import datasets
from verl.utils.hdfs_io import copy, makedirs


def main():
    parser = argparse.ArgumentParser(description="Preprocess aisale dataset to parquet format")
    parser.add_argument("--local_dir", default="./data/0515/data_human_baseline", 
                      help="Local directory to save parquet files")
    parser.add_argument("--hdfs_dir", default=None, 
                      help="HDFS directory to copy parquet files (optional)")
    parser.add_argument("--data_source", 
                      default="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/data/0515/data_human_baseline",
                      help="Path to source jsonl files")
    
    args = parser.parse_args()

    # 创建本地输出目录
    os.makedirs(args.local_dir, exist_ok=True)

    # 加载原始jsonl数据集
    ds = datasets.load_dataset("json", data_files={
        "train": os.path.join(args.data_source, "train.jsonl"),
        "val": os.path.join(args.data_source, "val.jsonl"),
        "test": os.path.join(args.data_source, "test.jsonl"),
    })

    # 定义数据处理函数：添加顶层question和answer字段
    def make_map_fn(split):
        def process_fn(example, idx):
            # 提取原始问题和答案
            question_raw = example.pop("instruction", "")
            answer_raw = example.pop("output", "")

            # 构造完整数据结构，包含顶层question和answer字段
            data = {
                # 顶层字段：直接对应训练配置中的prompt和response
                
                # 保留原有嵌套结构（不影响训练，可用于后续分析）
                "data_source": args.data_source,
                "prompt": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": question_raw}
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data
        return process_fn

    # 处理各拆分数据集
    train_dataset = ds["train"].map(make_map_fn("train"), with_indices=True)
    val_dataset = ds["val"].map(make_map_fn("val"), with_indices=True)
    test_dataset = ds["test"].map(make_map_fn("test"), with_indices=True)

    # 保存为parquet格式
    train_path = os.path.join(args.local_dir, "train_sft.parquet")
    val_path = os.path.join(args.local_dir, "val_sft.parquet")
    test_path = os.path.join(args.local_dir, "test_sft.parquet")

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)
    test_dataset.to_parquet(test_path)

    print(f"Parquet files saved to {args.local_dir}")
    print(f"Train: {train_path} ({len(train_dataset)} samples)")
    print(f"Val: {val_path} ({len(val_dataset)} samples)")
    print(f"Test: {test_path} ({len(test_dataset)} samples)")

    # 复制到HDFS（如果指定）
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"Files copied to HDFS: {args.hdfs_dir}")


if __name__ == "__main__":
    main()
