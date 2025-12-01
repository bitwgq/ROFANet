import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as prfs, confusion_matrix
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion, get_test_loaders,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np

import warnings
warnings.filterwarnings("ignore")

"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""
Set up environment: define paths, download data, and set device
"""

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

train_loader, val_loader = get_loaders(opt)
test_loader = get_test_loaders(opt)

"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')
model = load_model(opt, dev)

criterion = get_criterion(opt)

optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
best_test_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')
total_step = -1

for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()
    test_metrics = initialize_metrics()
    CM_train = 0
    CM_val = 0
    CM_test = 0
    """
    Begin Training
    """
    model.train()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels, OFF_L, batch_img_rd in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
        OFF_L = OFF_L.float().to(dev)
        batch_img_rd = batch_img_rd.float().to(dev)

        optimizer.zero_grad()

        cd_preds, off, off_d1, off_d2, A_aligned = model(batch_img1, batch_img2)

        i = 0
        temp_loss = 0.0
        for pred in cd_preds:
            if pred.size(2) != labels.size(2):
                temp_loss = temp_loss + criterion(pred, F.interpolate(labels, size=pred.size(2),mode="nearest"))
            else:
                temp_loss = temp_loss + criterion(pred, labels)
            i += 1
        G_loss = temp_loss
        off_loss = F.mse_loss(off, OFF_L) + F.mse_loss(off_d1, OFF_L-off) + F.mse_loss(off_d2, OFF_L-off-off_d1)
        apt_loss = F.l1_loss(A_aligned, batch_img_rd)
        loss = G_loss + off_loss*0.001 + apt_loss*0.01

        loss.backward()
        optimizer.step()

        _, cd_preds = torch.max(cd_preds[-1], 1)

        CM_NEW = confusion_matrix(labels.data.cpu().numpy().flatten(),
                                  cd_preds.data.cpu().numpy().flatten())
        if CM_NEW.size == 1:
            CM_NEW = np.atleast_2d(CM_NEW)
            CM_NEW = np.pad(CM_NEW, (0, 1), 'constant', constant_values=(0))
        CM_train = CM_train + CM_NEW

        del batch_img1, batch_img2, labels, OFF_L, batch_img_rd

    tp_sum = CM_train[1, 1]
    pred_sum = tp_sum + CM_train[0, 1]
    true_sum = tp_sum + CM_train[1, 0]
    tp_all = tp_sum + CM_train[0, 0]
    all = CM_train[0, 1] + CM_train[1, 0] + tp_all
    acc = tp_all/all
    pre = tp_sum/pred_sum
    rec = tp_sum/true_sum
    f1 = 2*pre*rec/(pre+rec)
    train_metrics = set_metrics(train_metrics,
                                loss,
                                acc,
                                pre,
                                rec,
                                f1,
                                scheduler.get_last_lr())
    scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(train_metrics))

    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels, OFF_L, batch_img_rd in val_loader:

            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)
            OFF_L = OFF_L.float().to(dev)
            batch_img_rd = batch_img_rd.float().to(dev)

            cd_preds, off, off_d1, off_d2, A_aligned = model(batch_img1, batch_img2)

            cd_loss = criterion(cd_preds[-1], labels)

            _, cd_preds = torch.max(cd_preds[-1], 1)

            CM_NEW = confusion_matrix(labels.data.cpu().numpy().flatten(),
                                      cd_preds.data.cpu().numpy().flatten())
            if CM_NEW.size == 1:
                CM_NEW = np.atleast_2d(CM_NEW)
                CM_NEW = np.pad(CM_NEW, (0, 1), 'constant', constant_values=(0))
            CM_val = CM_val + CM_NEW

            del batch_img1, batch_img2, labels, OFF_L, batch_img_rd

        """
        compute metrices
        """
        tp_sum = CM_val[1, 1]
        pred_sum = tp_sum + CM_val[0, 1]
        true_sum = tp_sum + CM_val[1, 0]
        tp_all = tp_sum + CM_val[0, 0]
        all = CM_val[0, 1] + CM_val[1, 0] + tp_all
        acc = tp_all / all
        pre = tp_sum / pred_sum
        rec = tp_sum / true_sum
        f1 = 2 * pre * rec / (pre + rec)

        val_metrics = set_metrics(val_metrics,
                                  cd_loss,
                                  acc,
                                  pre,
                                  rec,
                                  f1,
                                  scheduler.get_last_lr())

        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(val_metrics))

        """
        Store the weights of good epochs based on validation results
        """

        for batch_img1, batch_img2, labels, OFF_L, batch_img_rd in test_loader:
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)
            OFF_L = OFF_L.float().to(dev)
            batch_img_rd = batch_img_rd.float().to(dev)

            cd_preds, off, off_d1, off_d2, A_aligned = model(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds[-1], labels)

            _, cd_preds = torch.max(cd_preds[-1], 1)

            CM_NEW = confusion_matrix(labels.data.cpu().numpy().flatten(),
                                      cd_preds.data.cpu().numpy().flatten())
            if CM_NEW.size == 1:
                CM_NEW = np.atleast_2d(CM_NEW)
                CM_NEW = np.pad(CM_NEW, (0, 1), 'constant', constant_values=(0))
            CM_test = CM_test + CM_NEW

            del batch_img1, batch_img2, labels, OFF_L, batch_img_rd

        """
        compute metrices
        """
        tp_sum = CM_test[1, 1]
        pred_sum = tp_sum + CM_test[0, 1]
        true_sum = tp_sum + CM_test[1, 0]
        tp_all = tp_sum + CM_test[0, 0]
        all = CM_test[0, 1] + CM_test[1, 0] + tp_all
        acc = tp_all / all
        pre = tp_sum / pred_sum
        rec = tp_sum / true_sum
        f1 = 2 * pre * rec / (pre + rec)

        test_metrics = set_metrics(test_metrics,
                                  cd_loss,
                                  acc,
                                  pre,
                                  rec,
                                  f1,
                                  scheduler.get_last_lr())

        logging.info("EPOCH {} TEST METRICS".format(epoch) + str(test_metrics))
        if (val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']):
            logging.info('updata the model')
            metadata['validation_metrics'] = val_metrics

            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')

            torch.save(model, './tmp/best'+'.pt')
            best_metrics = val_metrics
        torch.save(model, './tmp/' + str(epoch) + '_' + str(f1) + '.pt')
        print('An epoch finished.')

writer.close()  # close tensor board
print('Train Done! The best result (val) ia:')
print(best_metrics)
