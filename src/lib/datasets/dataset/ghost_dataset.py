import os
import os.path as osp
from torch.utils.data import Dataset
import numpy as np

class GhostDataset(Dataset):
    def __init__(self, dataset_root, transforms):
        self.npz_paths = {}
        count = 0
        seqs = os.listdir(dataset_root)
        for seq in seqs:
            seq_path = osp.join(dataset_root, seq, 'preprocess')
            npzs = os.listdir(seq_path)
            for npz in npzs:
                npz_path = osp.join(seq_path, npz)
                self.npz_paths[count] = npz_path
                count += 1
        # import pdb; pdb.set_trace()
        self.transforms = transforms


    def __len__(self):
        return len(self.npz_paths)


    def __getitem__(self, idx):
        path = self.npz_paths[idx]
        data = np.load(path)


        # return self.transforms(Image.fromarray(data['track_feat'])), self.transforms(Image.fromarray(data['det_feat'])), data['target_delta_bbox']
        # return self.transforms(data['track_feat']), self.transforms(data['det_feat']), data['target_delta_bbox']


        track_feat = self.transforms(data['track_feat'])
        det_feat = self.transforms(data['det_feat'])


        track_tlbr = data['track_tlbr']
        det_tlbr = data['det_tlbr']
        tlwh_history = data['tlwh_history']
        target_delta_bbox = data['target_delta_bbox']

        # print(track_feat.shape, det_feat.shape)
        # print(track_tlbr.shape, det_tlbr.shape)
        # print(tlwh_history.shape)
        # print(target_delta_bbox.shape)

        track_tlbr[0] /= 1088
        track_tlbr[1] /= 608
        track_tlbr[2] /= 1088
        track_tlbr[3] /= 608

        det_tlbr[0] /= 1088
        det_tlbr[1] /= 608
        det_tlbr[2] /= 1088
        det_tlbr[3] /= 608

        tlwh_history[:,0] /= 1088
        tlwh_history[:,1] /= 608
        tlwh_history[:,2] /= 1088
        tlwh_history[:,3] /= 608

        target_delta_bbox[0] /= 1088
        target_delta_bbox[1] /= 608
        target_delta_bbox[2] /= 1088
        target_delta_bbox[3] /= 608

        return track_feat, det_feat, track_tlbr, det_tlbr, tlwh_history, target_delta_bbox
        # return track_feat, track_tlbr, tlwh_history, target_delta_bbox

        # return self.transforms(data['track_feat']), self.transforms(data['det_feat']),\
        #        data['track_tlbr'], data['det_tlbr'], data['tlwh_history'], data['target_delta_bbox']