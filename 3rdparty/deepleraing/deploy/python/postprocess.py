"""后处理层（Deploy-time Postprocessing）。

把 head 预测张量解码为最终检测结果（TransFusionHead bbox coder + NMS）。
与 C++ 侧 ``deploy/cpp/postprocess.*`` 逐层对应（Postprocessor 类 +
decode）。TRT head.bbox 输出 8 个预测 tensor + query_labels（图内 topk
的 proposal 类别 id，解码时转回 int 做 one_hot）。
"""

import torch

__all__ = ["Postprocessor"]


class Postprocessor:
    """后处理层：head 预测 -> (boxes, scores, labels)。

    职责边界：只做检测解码 / NMS / 过滤，不含模型前向。
    """

    def __init__(self, model):
        self.head = model.heads["object"]

    @torch.no_grad()
    def decode(self, preds, metas):
        """解码 head 预测 -> (boxes, scores, labels)。

        Args:
            preds: 9-key 预测 dict（8 个 pred + query_labels）。
            metas: collated 元信息（含 box_3d_limit / img_metas 等）。

        Returns:
            (boxes, scores, labels)：LibreBox3D / tensor、score、类别 id。
        """
        boxes, scores, labels = self.head.get_bboxes(([preds],), metas)[0]
        return boxes, scores, labels

    @torch.no_grad()
    def __call__(self, preds, metas):
        return self.decode(preds, metas)
