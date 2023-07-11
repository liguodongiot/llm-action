import os
import argparse
import mindspore as ms
from tqdm import tqdm
 
def merge_ckpt(args_opt):
    # Transform checkpoint directory
    # return str `save_checkpoint_path`
    print("args_opt.src_strategy_file: ", args_opt.src_strategy_file)
    rank_list = ms.rank_list_for_transform(0, args_opt.src_strategy_file, None)
    checkpoint_file_map = {}
    for rank_id in rank_list:
        checkpoint_file_map[rank_id] = os.path.join(args_opt.src_checkpoints_dir,
                                                    "checkpoint",
                                                    f"rank_{rank_id}",
                                                    f"glm-6b-lora_rank_{rank_id}-{args_opt.src_postfix}.ckpt")
    if not os.path.exists(args_opt.dst_checkpoints_dir):
        os.mkdir(args_opt.dst_checkpoints_dir)
    save_checkpoint_path = os.path.join(args_opt.dst_checkpoints_dir, "transformed.ckpt")
    print('checkpoint_file_map', checkpoint_file_map)
    print('save_checkpoint_path', save_checkpoint_path)
    ms.transform_checkpoint_by_rank(0, checkpoint_file_map, save_checkpoint_path,
                                    args_opt.src_strategy_file, None)
    print("transform ckpt done.")
    return save_checkpoint_path
 
def clean_ckpt(ms_ckpt_path):
    '''
    filter keyword `adam` in ckpt and generate new ckpt
    print layer name in `check_ms_ckpt.txt`
    '''
    print("Filtering ckpt, this may take a while.")
    ms_param = ms.load_checkpoint(ms_ckpt_path)
    new_param = []
    with open("check_ms_ckpt.txt", "w") as fp:
        for k, v in tqdm(ms_param.items(), ascii=True, ncols=120):
            if "adam" in k:
                continue
            fp.write(f"{k} {v.shape} {v.dtype}\n")
            new_param.append({"name": k, "data": ms.Tensor(v.numpy())})
    save_path, ckpt_name = os.path.split(ms_ckpt_path)
    ckpt_name = "filtered_" + ckpt_name
    ms_ckpt_path = os.path.join(save_path, ckpt_name)
    ms.save_checkpoint(new_param, ms_ckpt_path)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform checkpoint dir")
    parser.add_argument("--src_postfix", type=str, default="1790_4",
                        help="Source ckpt postfix str.")
    parser.add_argument("--src_checkpoints_dir", type=str, default="scripts",
                        help="The source checkpoint directory.")
    parser.add_argument("--src_strategy_file", type=str, default="ckpt_strategy.ckpt",
                        help="The source strategy file")
    parser.add_argument("--dst_checkpoints_dir", type=str, default="transformed_ckpt",
                        help="The destination checkpoint directory.")
 
    args_opt = parser.parse_args()
    save_checkpoint_path = merge_ckpt(args_opt)
    clean_ckpt(save_checkpoint_path)
