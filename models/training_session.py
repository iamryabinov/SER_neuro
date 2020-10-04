import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import time
import os
import shutil
from iemocap import *
from torchsummary import summary
from models import *



class TrainingSession():
    def __init__(self, model, dataset,
                 criterion, optimizer, scheduler, num_epochs, batch_size, device,
                 path_to_weights, path_to_results):
        print('INITIALIZING TRAINING SESSION...')
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.path_to_weights = path_to_weights
        self.path_to_results = path_to_results
        self.device = device
        self.criterion = criterion
        self.name = '{}__{}'.format(self.model.name, self.dataset.name)
        self.trainloader, self.testloader = train_test_loaders(self.dataset, validation_ratio=0.2, num_workers=0,
                                                               batch_size=self.batch_size)
        print('Loaders ready')
        print('TRAINING SESSION {} INITIALIZED'.format(self.name))
        checkpoint_file_name = '{}.pt'.format(self.name)
        print('Trying to load checkpoint from file')
        try:
            if os.listdir(path_to_weights) == []:
                raise FileNotFoundError
            for filename in os.listdir(path_to_weights):
                if checkpoint_file_name in filename:
                    print('Found file')
                    self.checkpoint_path = self.path_to_weights + filename
                    print('Loading file {}'.format(self.checkpoint_path))
                    epoch, self.results_dict = self.load_ckp()
                    self.current_epoch = epoch + 1
                    print('Success!')
        except FileNotFoundError:
            print('File not found, starting from scratch...')
            self.current_epoch = 1
            self.checkpoint_path = self.path_to_weights + checkpoint_file_name
            self.results_dict = {}

    def load_ckp(self):
        checkpoint = torch.load(self.checkpoint_path)
        print('Updating model_alex...')
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Updating optimizer_alex...')
        self.optimizer.load_state_dict(checkpoint['optimizer_alex'])
        return checkpoint['epoch'], checkpoint['results']

    def save_ckp(self, checkpoint, is_best):
        torch.save(checkpoint, self.checkpoint_path)
        if is_best:
            print('## Saving best model_alex')
            best_fpath = self.path_to_results + '{}__best_model.pt'.format(self.name)
            shutil.copyfile(self.checkpoint_path, best_fpath)

    def execute(self):

        if self.current_epoch == self.num_epochs + 1:
            print('===================TRAINING SESSION ENDED!===========================')
            return self.model, self.results_dict
        else:
            print('======================================================================')
            print('=============TRAINING SESSION STARTED AT EPOCH {}====================='.format(self.current_epoch))
            self.model, self.results_dict = self.training_loop(trainloader=self.trainloader,
                                                               testloader=self.testloader,
                                                               results_dict=self.results_dict)
        print('===================TRAINING SESSION ENDED!===========================')

    def training_loop(self, trainloader, testloader, results_dict):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        device = self.device
        model.to(device)
        starting_epoch = self.current_epoch
        ending_epoch = self.num_epochs + 1
        criterion = self.criterion
        dataset_size = len(trainloader.dataset)
        correct = 0
        total = 0
        best_acc = 0.0
        if starting_epoch == 1:
            train_acc_list = []
            val_acc_list = []
            train_loss_list = []
            val_loss_list = []
        else:
            train_acc_list = results_dict['train accuracy']
            val_acc_list = results_dict['test accuracy']
            train_loss_list = results_dict['train loss']
            val_loss_list = results_dict['test loss']
        # iterate over epochs
        for epoch_num in range(starting_epoch, ending_epoch):
            print('======================================================================')
            print('Epoch #%d' % epoch_num)
            epoch_loss = 0.0
            model.train()
            t0 = time.time()
            # iterate over batches
            for i, (data, target) in enumerate(trainloader):
                print(i)
                data = data.to(device)
                target = target[0]
                target = target.to(device)
                optimizer.zero_grad()  # zero all the gradient tensors
                predicted = model(data)  # run forward step
                loss = criterion(predicted, target)  # compute loss
                loss.backward()  # compute gradient tensors
                optimizer.step()  # update parameters
                epoch_loss += loss.item() * data.size(0)  # compute the training loss value
                total += target.size(0)
                _, pred_labels = torch.max(predicted.data, 1)
                correct += (pred_labels == target).sum().item()
            t = time.time() - t0
            epoch_loss /= dataset_size
            train_acc = correct / total
            print('# Time passed: %.0f s' % t)
            print('# Epoch loss = %.4f' % epoch_loss)
            print('# Train acc = {}'.format(train_acc))
            print('# Validation process on validation set')
            val_loss, val_acc = self.validate(model, testloader, device)
            scheduler.step(val_loss)
            print('# Validation loss = {}'.format(val_loss))
            print('# Validation acc = {}'.format(val_acc))
            if val_acc > best_acc:
                is_best = True
                best_acc = val_acc
            else:
                is_best = False
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            train_loss_list.append(epoch_loss)
            val_loss_list.append(val_loss)
            results_dict = {
                'train accuracy': train_acc_list,
                'test accuracy': val_acc_list,
                'train loss': train_loss_list,
                'test loss': val_loss_list
            }
            checkpoint = {
                'epoch': epoch_num,
                'state_dict': model.state_dict(),
                'optimizer_alex': optimizer.state_dict(),
                'results': results_dict
            }
            self.model = model
            self.optimizer = optimizer
            print('# Saving checkpoint...')
            self.save_ckp(checkpoint, is_best)
            print('# Done and done!')
        return model, results_dict

    def validate(self, model, testloader, device):
        dataset_size = len(testloader.dataset)
        correct = 0
        total = 0
        model.eval()
        epoch_loss = 0.0
        for i, (data, target) in enumerate(testloader):
            t0 = time.time()
            data = data.to(device)
            target = target[0]
            target = target.to(device)
            with torch.no_grad():
                # run forward step
                predicted = model(data)
                loss = self.criterion(predicted, target)
                epoch_loss += loss.item() * data.size(0)
            _, pred_labels = torch.max(predicted.data, 1)
            total += target.size(0)
            correct += (pred_labels == target).sum().item()
        return epoch_loss / dataset_size, correct / total

    def overfit_one_batch(self, num_epochs=100, batch_size=10):
        for epoch_num in range(num_epochs):
            print('======================================================================')
            print('Epoch #%d' % epoch_num)
            epoch_loss = 0.0
            self.model.train()
            t0 = time.time()
            trainloader, _ = train_test_loaders(self.dataset, batch_size=batch_size)
            first_batch = next(iter(trainloader))
            dataset_size = batch_size * 50
            total = 0
            correct = 0
            for batch_idx, (data, target) in enumerate([first_batch] * 50):
                print(batch_idx)
                data = data.to(self.device)
                target = target[0]
                target = target.to(self.device)
                self.optimizer.zero_grad()  # zero all the gradient tensors
                predicted = self.model(data)  # run forward step
                loss = self.criterion(predicted, target)  # compute loss
                print('loss = {}'.format(loss))
                loss.backward()  # compute gradient tensors
                self.optimizer.step()  # update parameters
                epoch_loss += loss.item() * data.size(0)  # compute the training loss value
                total += target.size(0)
                _, pred_labels = torch.max(predicted.data, 1)
                correct += (pred_labels == target).sum().item()
            t = time.time() - t0
            epoch_loss /= dataset_size
            train_acc = correct / total
            self.scheduler.step(epoch_loss)
            print('# Time passed: %.0f s' % t)
            print('# Epoch loss = %.4f' % epoch_loss)
            print('# Train acc = {}'.format(train_acc))
            print('# Validation process on validation set')




if __name__ == '__main__':
    pass