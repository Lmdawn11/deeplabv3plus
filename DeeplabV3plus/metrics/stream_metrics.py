import numpy as np
from sklearn.metrics import confusion_matrix


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self, auc, f1):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        mean_auc = auc
        mean_f1 = f1
        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Mean AUC": mean_auc,
            "Mean F1:": mean_f1
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


# def get_dataset(opts):
#     """ Dataset And Augmentation
#     """
#     if opts.dataset == 'voc':
#         train_transform = et.ExtCompose([
#             # et.ExtResize(size=opts.crop_size),
#             et.ExtRandomScale((0.5, 2.0)),
#             et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
#             et.ExtRandomHorizontalFlip(),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#         if opts.crop_val:
#             val_transform = et.ExtCompose([
#                 et.ExtResize(opts.crop_size),
#                 et.ExtCenterCrop(opts.crop_size),
#                 et.ExtToTensor(),
#                 et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#             ])
#         else:
#             val_transform = et.ExtCompose([
#                 et.ExtToTensor(),
#                 et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#             ])
#         # train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
#         #                             image_set='train', download=opts.download, transform=train_transform)
#         # val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
#         #                           image_set='val', download=False, transform=val_transform)
#
#     # if opts.dataset == 'cityscapes':
#     #     train_transform = et.ExtCompose([
#     #         # et.ExtResize( 512 ),
#     #         et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
#     #         et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
#     #         et.ExtRandomHorizontalFlip(),
#     #         et.ExtToTensor(),
#     #         et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#     #                         std=[0.229, 0.224, 0.225]),
#     #     ])
#     #
#     #     val_transform = et.ExtCompose([
#     #         # et.ExtResize( 512 ),
#     #         et.ExtToTensor(),
#     #         et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#     #                         std=[0.229, 0.224, 0.225]),
#     #     ])
#     #
#     #     train_dst = Cityscapes(root=opts.data_root,
#     #                            split='train', transform=train_transform)
#     #     val_dst = Cityscapes(root=opts.data_root,
#     #                          split='val', transform=val_transform)
#     return train_dst, val_dst
def extractGTs(gt, erodeKernSize=15, dilateKernSize=11):
    from scipy.ndimage import minimum_filter, maximum_filter
    gt1 = minimum_filter(gt, erodeKernSize)
    gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
    return gt0, gt1


def computeMetricsContinue(values, gt0, gt1):
    values = values.flatten().astype(np.float32)
    gt0 = gt0.flatten().astype(np.float32)
    gt1 = gt1.flatten().astype(np.float32)

    inds = np.argsort(values)
    inds = inds[(gt0[inds] + gt1[inds]) > 0]
    vet_th = values[inds]
    gt0 = gt0[inds]
    gt1 = gt1[inds]

    TN = np.cumsum(gt0)
    FN = np.cumsum(gt1)
    FP = np.sum(gt0) - TN
    TP = np.sum(gt1) - FN

    msk = np.pad(vet_th[1:] > vet_th[:-1], (0, 1), mode='constant', constant_values=True)
    FP = FP[msk]
    TP = TP[msk]
    FN = FN[msk]
    TN = TN[msk]
    vet_th = vet_th[msk]

    return FP, TP, FN, TN, vet_th


def computeF1(FP, TP, FN, TN):
    return 2 * TP / np.maximum((2 * TP + FN + FP), 1e-32)


def computeMetrics_th(values, gt, gt0, gt1, th):
    values = values > th
    values = values.flatten().astype(np.uint8)
    gt = gt.flatten().astype(np.uint8)
    gt0 = gt0.flatten().astype(np.uint8)
    gt1 = gt1.flatten().astype(np.uint8)

    gt = gt[(gt0 + gt1) > 0]
    values = values[(gt0 + gt1) > 0]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt, values, labels=[0, 1])

    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TP = cm[1, 1]

    return FP, TP, FN, TN


def computeLocalizationMetrics(map, gt):
    gt0, gt1 = extractGTs(gt)

    # best threshold
    try:
        FP, TP, FN, TN, _ = computeMetricsContinue(map, gt0, gt1)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_best = max(np.max(f1), np.max(f1i))
    except:
        import traceback
        traceback.print_exc()
        F1_best = np.nan

    # fixed threshold
    try:
        FP, TP, FN, TN = computeMetrics_th(map, gt, gt0, gt1, 0.5)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_th = max(f1, f1i)
    except:
        import traceback
        traceback.print_exc()
        F1_th = np.nan

    return F1_best, F1_th
