import torch.nn as nn
from .alexnet import alexnet
from .resnet import resnet18
import torch

class GPN(nn.Module):
    def __init__(self, network='alexnet'):
        super(GPN, self).__init__()

        self.network = network
        feat_dim = 256
        self.extractor = None
        if network == 'alexnet':
            self.extractor = alexnet(pretrained=True)
            feat_dim = 256
        elif network == 'resnet':
            self.extractor = resnet18(pretrained=True)
            feat_dim = 512

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2 * feat_dim * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4),
        )

        # TODO: add classification layer and CE loss


        # for param in self.extractor.parameters():
        #     param.requires_grad = False

        self.vis_thres = 0.1
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm_reg = nn.Linear(32, 4)

        self.dropout = nn.Dropout()

    def forward(self, track_imgs, det_imgs, track_tlbrs, det_tlbrs, tlwh_histories):
        """
        track_imgs: bs, 3, h, w
        det_imgs: bs, 3, h, w
        track_tlbrs: bs, 4
        det_tlbrs: bs, 4
        bbox_histories: bs, history_size, 4
        """

        if self.network == 'lstm':
            track_tlwhs = track_tlbrs
            track_tlwhs[:,2:4] -= track_tlwhs[:,0:2]

            _, (ht, _) = self.lstm(tlwh_histories)
            ht = self.dropout(ht)
            lstm_bboxes = self.lstm_reg(ht[-1])

            delta_bbox = torch.zeros(track_tlbrs.size()).cuda()
            bs = track_imgs.size(0)
            for i in range(bs):
                if True:
                    delta_bbox[i] = lstm_bboxes[i]


            return delta_bbox

        elif self.network == 'alexnet':

            track_feat, track_featmap = self.extractor(track_imgs) # track_featmap: 1, 256, 6, 6
            det_feat, det_featmap = self.extractor(det_imgs)
            concatenated_feat = torch.cat((track_featmap, det_featmap), dim=1) # 1, 512, 6, 6
            x = torch.flatten(concatenated_feat, 1)
            delta_bbox = self.classifier(x)

            track_tlwhs = track_tlbrs
            det_tlwhs = det_tlbrs
            track_tlwhs[:,2:4] -= track_tlwhs[:,0:2]
            det_tlwhs[:,2:4] -= det_tlwhs[:,0:2]
            visibility = 1 - bbox_overlaps(
                np.ascontiguousarray(track_tlwhs.cpu().detach().numpy(), dtype=np.float),
                np.ascontiguousarray(det_tlwhs.cpu().detach().numpy(), dtype=np.float)
            )

            _, (ht, _) = self.lstm(tlwh_histories)
            ht = self.dropout(ht)
            lstm_bboxes = self.lstm_reg(ht[-1])
            cnn_bboxes = track_tlwhs + delta_bbox
            # import pdb; pdb.set_trace()

            delta_bbox = torch.zeros(track_tlbrs.size()).cuda()
            bs = track_imgs.size(0)
            for i in range(bs):
                vis_i = visibility[i,i]
                if vis_i < self.vis_thres:
                # if True:
                # if False:
                    delta_bbox[i] = lstm_bboxes[i]
                else:
                    delta_bbox[i] = lstm_bboxes[i] * (1 - vis_i) + cnn_bboxes * vis_i

            return delta_bbox

        elif self.network == 'resnet':

            track_feat, track_featmap = self.extractor(track_imgs) # track_featmap: 1, 128, 28, 28
            det_feat, det_featmap = self.extractor(det_imgs)
            concatenated_feat = torch.cat((track_featmap, det_featmap), dim=1) # 1, 512, 7, 7
            x = torch.flatten(concatenated_feat, 1)
            delta_bbox = self.classifier(x)


            track_tlwhs = track_tlbrs
            det_tlwhs = det_tlbrs
            track_tlwhs[:,2:4] -= track_tlwhs[:,0:2]
            det_tlwhs[:,2:4] -= det_tlwhs[:,0:2]
            visibility = 1 - bbox_overlaps(
                np.ascontiguousarray(track_tlwhs.cpu().detach().numpy(), dtype=np.float),
                np.ascontiguousarray(det_tlwhs.cpu().detach().numpy(), dtype=np.float)
            )

            _, (ht, _) = self.lstm(tlwh_histories)
            ht = self.dropout(ht)
            lstm_bboxes = self.lstm_reg(ht[-1])
            cnn_bboxes = track_tlwhs + delta_bbox
            # import pdb; pdb.set_trace()

            delta_bbox = torch.zeros(track_tlbrs.size()).cuda()
            bs = track_imgs.size(0)
            for i in range(bs):
                vis_i = visibility[i,i]
                if vis_i < self.vis_thres:
                # if True:
                # if False:
                    delta_bbox[i] = lstm_bboxes[i]
                else:
                    delta_bbox[i] = lstm_bboxes[i] * (1 - vis_i) + cnn_bboxes * vis_i

            return delta_bbox