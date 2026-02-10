# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tarfile
import torch
import webdataset as wds
import tarfile
import io
import os
import imageio
# import mmcv
import re
import random
import cv2
def go_image_tar_decoder(sample, pipeline, sample_num, interval, size=256):
    # result = dict(__key__=sample["__key__"])
    # import pdb;pdb.set_trace()
    # for key, value in sample.items():
    #     if not key.endswith(".png"):
    invalid = False
    value = sample['tar']
    with tarfile.open(fileobj=io.BytesIO(value), mode="r:") as inner_tar:
        inner_tar = tarfile.open(fileobj=io.BytesIO(value), mode="r:")
        all_move_names = []
        for inner_member in inner_tar.getmembers():
            if inner_member.isfile():
                all_move_names.append(inner_member.name.split('/')[-1])
            else:
                file_path = inner_member.name
                _idx = file_path.split('/')[-1]
        # if len(all_move_names) == 1:
        #     import pdb;pdb.set_trace()
        if len(all_move_names) < sample_num:
            invalid = True
        extra_move_names = [move_name for move_name in all_move_names if 'extra' in move_name]
        ignore_move_names = [move_name.replace('i_', '') for move_name in all_move_names if 'i_' in move_name]
        ignore_move_names = sorted(ignore_move_names, key=lambda x: int(re.search(r'\d+', x).group()))
        max_ignore_idx = len(ignore_move_names)
        move_names = [move_name.replace('i_', '') for move_name in all_move_names if 'extra' not in move_name]
        move_names = sorted(move_names, key=lambda x: int(re.search(r'\d+', x).group()))
        try:
            start_move = int(move_names[max_ignore_idx].replace('.png', ''))
        except:
            start_move = int(move_names[0].replace('.png', ''))
        move_names = [file_path+'/'+move_name for move_name in move_names]
        black_move_start = start_move % 2 == 0
        if black_move_start and start_move != 0:
            # import pdb;pdb.set_trace()
            # move_names = move_names[(max_ignore_idx+1):]
            max_ignore_idx += 1
            black_move_idx = [i for i in range(len(move_names)-max_ignore_idx) if i % 2 == 1]
        elif black_move_start and start_move == 0:
            empty = ["./data/go_dataset_size9/empty_board.png"]
            empty.extend(move_names)
            move_names = empty
            black_move_idx = [i for i in range(len(move_names)-max_ignore_idx) if i % 2 == 1]
        else:
            black_move_idx = [i for i in range(len(move_names)-max_ignore_idx) if i % 2 == 1]
        if len(black_move_idx) == 0:
            # import pdb;pdb.set_trace()
            move_names = [move_name.replace('i_', '') for move_name in all_move_names if 'extra' not in move_name]
            move_names = sorted(move_names, key=lambda x: int(re.search(r'\d+', x).group()))
            move_names = [file_path+'/'+move_name for move_name in move_names]
            sample_black_idx = min(max_ignore_idx, len(ignore_move_names))
            # print('--------empty black--------')
        else:
            sample_black_idx= random.sample(black_move_idx, 1)[0]
            sample_black_idx = sample_black_idx + max_ignore_idx

        if sample_num == 2:
            
            try:
                name2 = move_names[sample_black_idx] if 'extra_'+move_names[sample_black_idx].split('/')[-1] not in extra_move_names else move_names[sample_black_idx].replace(_idx+'/', _idx+'/extra_')          
            except:
                sample_black_idx = 0
                name2 = move_names[sample_black_idx] if 'extra_'+move_names[sample_black_idx].split('/')[-1] not in extra_move_names else move_names[sample_black_idx].replace(_idx+'/', _idx+'/extra_')          
            # if int(name2.split('/')[-1].replace('.png', '').replace('extra_', '')) % 2 == 1:
            #     print(2)
            
            for _interval in range(interval, 0, -1):
                if _interval <= sample_black_idx:
                    break
            before_num = _interval
            # print(before_num)
            name1 = move_names[sample_black_idx-before_num] if not invalid else "./data/go_dataset_size9/empty_board.png"
            if sample_black_idx-before_num < len(ignore_move_names):
                name1 = name1.replace(_idx+'/', _idx+'/i_')
            try:
                file2_content = inner_tar.extractfile(name2).read()
                image2 = imageio.imread(io.BytesIO(file2_content))
            except:
                # import pdb;pdb.set_trace()
                print("--------Can't open image2--------", inner_tar.getmembers(), name2)
                image1 = cv2.imread('./data/go_dataset_size9/empty_board.png')
                image2 = cv2.imread('./data/go_dataset_size9/empty_board.png')
                invalid = True
            if 'empty' not in name1:
                try:
                    file1_content = inner_tar.extractfile(name1).read()
                    image1 = imageio.imread(io.BytesIO(file1_content))
                except:
                    # import pdb;pdb.set_trace()
                    print("--------Can't open image1--------", inner_tar.getmembers(), name1)
                    image1 = cv2.imread('./data/go_dataset_size9/empty_board.png')
                    image2 = cv2.imread('./data/go_dataset_size9/empty_board.png')
                    invalid = True
            else:
                image1 = cv2.imread('./data/go_dataset_size9/empty_board.png')
                image2 = cv2.imread('./data/go_dataset_size9/empty_board.png')
            
            image1, image2 = image1[:, :, :3], image2[:, :, :3]

            file_name = file_path.split('/')[-1]
            os.system(f'mkdir -p /opt/tiger/mmagicinit/train_data/{file_name}')
            cv2.imwrite(f'/opt/tiger/mmagicinit/train_data/{file_name}/0.png', image1[:, :, ::-1])
            cv2.imwrite(f'/opt/tiger/mmagicinit/train_data/{file_name}/1.png', image2[:, :, ::-1])

            if image1.shape[0] != size or image1.shape[1] != size:
                image1 = cv2.resize(image1, (size, size))
            if image2.shape[0] != size or image2.shape[1] != size:
                image2 = cv2.resize(image2, (size, size))
            imgs = [image1, image2]
        else:
            
            imgs = []
            if sample_num-1+max_ignore_idx > sample_black_idx and sample_num-1+max_ignore_idx < len(move_names):
                sample_black_idx = sample_num-1+max_ignore_idx
            # sample_black_idx = max(sample_black_idx, sample_num-1+max_ignore_idx)
            # print("_____", sample_black_idx, "_____")
            for fi in range(sample_num):
                idx = sample_black_idx-sample_num+fi+1
                name = move_names[idx] if idx >= 0 and idx < len(move_names) else move_names[0]
                # if fi == 0 and int(name.split('/')[-1].replace('.png', '')) % 2 == 0:
                #     import pdb;pdb.set_trace()
                if idx < len(ignore_move_names):
                    # import pdb;pdb.set_trace()
                    name = name.replace(_idx+'/', _idx+'/i_')
                if not(idx >= 0 and idx < len(move_names)):
                    # import pdb;pdb.set_trace()
                    print("--------Error idx--------", sample_black_idx, len(move_names))
         
                try:
                    file_content = inner_tar.extractfile(name).read()
                    image = imageio.imread(io.BytesIO(file_content))
                except:
                    print("--------Can't open image--------", inner_tar.getmembers(), name)
                    image = cv2.imread('./data/go_dataset_size9/empty_board.png')[:, :, ::-1]
                # else:
                #     image = cv2.imread('./data/go_dataset_size9/empty_board.png')[:, :, ::-1]
                image = image[:, :, :3]
                if image.shape[0] != size or image.shape[1] != size:
                    image = cv2.resize(image, (size, size))
                imgs.append(image)
    item = {}
    
    item['imgs'] = imgs
    item['original_shape'] = imgs[0].shape[:2]
    item['img_shape'] = imgs[0].shape[:2]
    return pipeline(item)
    # return result

if __name__ == '__main__':
    from mmengine.registry import build_from_cfg
    from ldm.registry import TRANSFORMS
    from torchvision.transforms import Compose
    from functools import partial
    dataset = wds.WebDataset('data/go_dataset_size9/kataselfpaly_filterdup_ignore_wcap_AI_image_tar/{000000000..000001259}.tar')
    # dataset = wds.WebDataset('data/go_dataset_size9/kataselfpaly_filterdup_ignore_wcap_10M_image_tar/{000000000..000001259}.tar').shuffle(1000)
    
    train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    # dict(type='SampleFrames', clip_len=17, frame_interval=2, num_clips=1),
    # dict(type='DecordDecode'),
    # dict(type='CenterCropLongEdgeVideo'),
    # dict(type='ResizeVideo', scale=(128, 128),keep_ratio=False),
    dict(type='PackVideoInputs')
    ]
    # pipeline_cfg = dataset_cfg.get('pipeline', False)
    pipeline = [build_from_cfg(p, TRANSFORMS) for p in train_pipeline]
    pipeline = Compose(pipeline)
    dataset = dataset.map(partial(go_image_tar_decoder, pipeline=pipeline, sample_num=2, interval=3))
    # dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=16)
    for x in dataset:
        pass
        # import pdb;pdb.set_trace()
        # print(1)
# with tarfile.open('/mnt/bn/zhicheng-dev-v6/dataset/go_dataset_size9/kataselfpaly_filterdup_ignore_wcap_10M_image/kataselfpaly_filterdup_ignore_wcap_10M_image_875-end.tar.gz', 'r') as tar:
#     # 获取tar包中所有的成员
#     members = tar.getmembers()
    
#     print(members)