
from coco_dataset import cocoid_to_name
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from torch import nn
import matplotlib.pyplot as plt


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

'''
def post_process(outputs, target_sizes):
    def center_to_corners_format(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    out_logits, boxes = outputs.logits, outputs.pred_boxes
    prob = nn.functional.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    print("OTTO1    ", boxes)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    print("OTTO2    ", boxes)
    # convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(boxes)
    print("OTTO3    ", boxes)
    results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
    return results
'''

def plot_results(pil_img, label, boxes, scores, sen):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, (xmin, ymin, xmax, ymax), c in zip(scores, boxes.tolist(), colors):
        if score > sen:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
            text = ""
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


class COCO_Scanner:
    def __init__(self, sensitivity = 0.9, categories = [18, 19, 20, 21, 22, 23, 25]):
        super(COCO_Scanner, self).__init__()
        self.fe = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.mod = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.s = sensitivity
        self.categories = categories

    def __call__(self, pil_img):
        scan_data = []
        inputs = self.fe(images=pil_img, return_tensors="pt")
        outputs = self.mod(**inputs)
        target_sizes = torch.tensor([pil_img.size[::-1]])
        results = self.fe.post_process(outputs, target_sizes=target_sizes)[0]
        m = 0.0
        l = None
        plot_results(pil_img, "", results["boxes"], results["scores"], self.s)
        for ir, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            box = box.tolist()
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            label = label.item()
            if label in cocoid_to_name:
                labelstr = "%d (%s)" % (label, cocoid_to_name[label])
            else:
                labelstr = str(label)
            if score > m:
                m = score
                l = label
            if score > self.s:
                if label in self.categories:
                    print("   Scanned: %s. Score: %f." % (labelstr, score))
                    cata = self.categories.index(label) + 1
                    scan_data.append({"box": box, "cata": cata})
        scan_data = sorted(scan_data, key=lambda x: (x["box"][0], x["box"][1]))
        return scan_data
