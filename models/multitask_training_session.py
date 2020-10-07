import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import time
import os
import shutil
from datasets.iemocap import *
from torchsummary import summary
from models.models_multi_task import *
from models.models_one_task import *
from torch.utils.data import DataLoader


class TrainingSession():
    def __init__(self, name, model, train_dataset,
                 criterion, optimizer, num_epochs, batch_size, device,
                 path_to_weights, path_to_results, test_dataset=None, ):
        print('INITIALIZING TRAINING SESSION...')
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.path_to_weights = path_to_weights
        self.path_to_results = path_to_results
        self.device = device
        self.criterion = criterion
        self.name = '{}__{}'.format(name, self.train_dataset.name)
        if self.test_dataset == None:
            self.trainloader, self.testloader = train_test_loaders(self.train_dataset, validation_ratio=0.2,
                                                                   num_workers=4, batch_size=self.batch_size)
        else:
            self.trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            self.testloader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
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
        device = self.device
        model.to(device)
        starting_epoch = self.current_epoch
        ending_epoch = self.num_epochs + 1
        criterion = self.criterion
        dataset_size = len(trainloader.dataset)
        correct_emotion = 0
        correct_speaker = 0
        correct_gender = 0
        total = 0
        best_acc = 0.0
        if starting_epoch == 1:
            train_acc_emotion_list = []
            train_acc_speaker_list = []
            train_acc_gender_list = []

            val_acc_emotion_list = []
            val_acc_speaker_list = []
            val_acc_gender_list = []

            train_loss_emotion_list = []
            train_loss_speaker_list = []
            train_loss_gender_list = []

            val_loss_emotion_list = []
            val_loss_speaker_list = []
            val_loss_gender_list = []
        else:
            train_acc_emotion_list = results_dict['emotion']['train_accuracy']
            train_acc_speaker_list = results_dict['speaker']['train_accuracy']
            train_acc_gender_list = results_dict['gender']['train_accuracy']

            val_acc_emotion_list = results_dict['emotion']['test_accuracy']
            val_acc_speaker_list = results_dict['speaker']['test_accuracy']
            val_acc_gender_list = results_dict['gender']['test_accuracy']

            train_loss_emotion_list = results_dict['emotion']['train_loss']
            train_loss_speaker_list = results_dict['speaker']['train_loss']
            train_loss_gender_list = results_dict['gender']['train_loss']

            val_loss_emotion_list = results_dict['emotion']['test_loss']
            val_loss_speaker_list = results_dict['speaker']['test_loss']
            val_loss_gender_list = results_dict['gender']['test_loss']

        # iterate over epochs
        for epoch_num in range(starting_epoch, ending_epoch):
            print('======================================================================')
            print('Epoch #%d' % epoch_num)
            epoch_loss_emotion = 0.0
            epoch_loss_speaker = 0.0
            epoch_loss_gender = 0.0
            epoch_loss_total = 0.0
            model.train()
            t0 = time.time()
            # iterate over batches
            for i, (data, target) in enumerate(trainloader):
                data = data.to(device)
                target_emotion, target_speaker, target_gender = target
                target_emotion = target_emotion.to(device)
                target_speaker = target_speaker.to(device)
                target_gender = target_gender.to(device)
                optimizer.zero_grad()  # zero all the gradient tensors
                predicted_emotion, predicted_speaker, predicted_gender = model(data)  # run forward step

                loss_emotion = criterion(predicted_emotion, target_emotion)  # compute loss
                loss_speaker = criterion(predicted_speaker, target_speaker)
                loss_gender = criterion(predicted_gender, target_gender)
                loss_total = loss_emotion + loss_speaker + loss_gender
                loss_total.backward()  # compute gradient tensors

                optimizer.step()  # update parameters

                epoch_loss_emotion += loss_emotion.item() * data.size(0)
                epoch_loss_speaker += loss_speaker.item() * data.size(0)
                epoch_loss_gender += loss_gender.item() * data.size(0)
                epoch_loss_total += loss_total.item() * data.size(0)  # compute the training loss value

                total += target[0].size(0)
                _, pred_labels_emotion = torch.max(predicted_emotion.data, 1)
                _, pred_labels_speaker = torch.max(predicted_speaker.data, 1)
                _, pred_labels_gender = torch.max(predicted_gender.data, 1)

                correct_emotion += (pred_labels_emotion == target_emotion).sum().item()
                correct_speaker += (pred_labels_speaker == target_speaker).sum().item()
                correct_gender += (pred_labels_gender == target_gender).sum().item()
            t = time.time() - t0

            epoch_loss_emotion /= dataset_size
            epoch_loss_gender /= dataset_size
            epoch_loss_speaker /= dataset_size

            train_acc_emotion = correct_emotion / total
            train_acc_speaker = correct_speaker / total
            train_acc_gender = correct_gender / total

            print('# Time passed: %.0f s' % t)
            print('# Epoch losses | emotion = {:.4f} | speaker = {:.4f} | gender = {:.4f} | total = {:.4f} |'.format(
                epoch_loss_emotion, epoch_loss_speaker, epoch_loss_gender, epoch_loss_total
            ))
            print('# Train accuracies | emotion = {} | speaker = {} | gender = {} |'.format(
                train_acc_emotion, train_acc_speaker, train_acc_gender
            ))
            print('# Validation process on validation set')
            val_losses, val_accuracies = self.validate(model, testloader, device)
            val_loss_emotion, val_loss_speaker, val_loss_gender, val_loss_total = val_losses
            val_acc_emotion, val_acc_speaker, val_acc_gender = val_accuracies
            print(
                '# Validation losses | emotion = {:.4f} | speaker = {:.4f} | gender = {:.4f} | total = {:.4f} |'.format(
                    val_loss_emotion, val_loss_speaker, val_loss_gender, val_loss_total
                ))
            print('# Validation accuracies | emotion = {} | speaker = {} | gender = {} |'.format(
                val_acc_emotion, val_acc_speaker, val_acc_gender
            ))
            if val_acc_emotion > best_acc:
                is_best = True
                best_acc = val_acc_emotion
            else:
                is_best = False

            train_acc_emotion_list.append(train_acc_emotion)
            train_acc_speaker_list.append(train_acc_speaker)
            train_acc_gender_list.append(train_acc_gender)

            val_acc_emotion_list.append(val_acc_emotion)
            val_acc_speaker_list.append(val_acc_speaker)
            val_acc_gender_list.append(val_acc_gender)

            train_loss_emotion_list.append(epoch_loss_emotion)
            train_loss_speaker_list.append(epoch_loss_speaker)
            train_loss_gender_list.append(epoch_loss_gender)

            val_loss_emotion_list.append(val_loss_emotion)
            val_loss_speaker_list.append(val_loss_speaker)
            val_loss_gender_list.append(val_loss_gender)

            results_dict_emotion = {
                'train accuracy': train_acc_emotion_list,
                'test accuracy': val_acc_emotion_list,
                'train loss': train_loss_emotion_list,
                'test loss': val_loss_emotion_list
            }

            results_dict_speaker = {
                'train accuracy': train_acc_speaker_list,
                'test accuracy': val_acc_speaker_list,
                'train loss': train_loss_speaker_list,
                'test loss': val_loss_speaker_list
            }

            results_dict_gender = {
                'train accuracy': train_acc_gender_list,
                'test accuracy': val_acc_gender_list,
                'train loss': train_loss_gender_list,
                'test loss': val_loss_gender_list
            }

            results_dict = {
                'emotion': results_dict_emotion,
                'speaker': results_dict_speaker,
                'gender': results_dict_gender
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
        model.eval()
        val_loss_emotion = 0.0
        val_loss_speaker = 0.0
        val_loss_gender = 0.0
        val_loss_total = 0.0
        correct_emotion = 0
        correct_speaker = 0
        correct_gender = 0
        total = 0
        for i, (data, target) in enumerate(testloader):
            data = data.to(device)
            target_emotion, target_speaker, target_gender = target
            target_emotion = target_emotion.to(device)
            target_speaker = target_speaker.to(device)
            target_gender = target_gender.to(device)
            with torch.no_grad():
                # run forward step
                predicted_emotion, predicted_speaker, predicted_gender = model(data)

                loss_emotion = self.criterion(predicted_emotion, target_emotion)  # compute loss
                loss_speaker = self.criterion(predicted_speaker, target_speaker)
                loss_gender = self.criterion(predicted_gender, target_gender)
                loss_total = loss_emotion + loss_speaker + loss_gender

                val_loss_emotion += loss_emotion.item() * data.size(0)
                val_loss_speaker += loss_speaker.item() * data.size(0)
                val_loss_gender += loss_gender.item() * data.size(0)
                val_loss_total += loss_total.item() * data.size(0)  # compute the training loss value

            total += target[0].size(0)
            _, pred_labels_emotion = torch.max(predicted_emotion.data, 1)
            _, pred_labels_speaker = torch.max(predicted_speaker.data, 1)
            _, pred_labels_gender = torch.max(predicted_gender.data, 1)

            correct_emotion += (pred_labels_emotion == target_emotion).sum().item()
            correct_speaker += (pred_labels_speaker == target_speaker).sum().item()
            correct_gender += (pred_labels_gender == target_gender).sum().item()

            val_loss_emotion /= dataset_size
            val_loss_gender /= dataset_size
            val_loss_speaker /= dataset_size
            val_loss_total += loss_total.item() * data.size(0)

            val_acc_emotion = correct_emotion / total
            val_acc_speaker = correct_speaker / total
            val_acc_gender = correct_gender / total

            val_losses = val_loss_emotion, val_loss_speaker, val_loss_gender, val_loss_total
            val_accuracies = val_acc_emotion, val_acc_speaker, val_acc_gender

        return val_losses, val_accuracies

    def overfit_one_batch(self, num_epochs=100, batch_size=10):
        model = self.model
        model.to(self.device)
        for epoch_num in range(num_epochs):
            print('======================================================================')
            print('Epoch #%d' % epoch_num)
            epoch_loss_emotion = 0.0
            epoch_loss_speaker = 0.0
            epoch_loss_gender = 0.0
            epoch_loss_total = 0.0
            t0 = time.time()
            self.model.train()
            t0 = time.time()
            trainloader, _ = train_test_loaders(self.train_dataset, batch_size=batch_size)
            first_batch = next(iter(trainloader))
            dataset_size = batch_size * 50
            total = 0
            correct_emotion = 0
            correct_speaker = 0
            correct_gender = 0
            for batch_idx, (data, target) in enumerate([first_batch] * 50):
                data = data.to(self.device)
                target_emotion, target_speaker, target_gender = target
                target_emotion = target_emotion.to(self.device)
                target_speaker = target_speaker.to(self.device)
                target_gender = target_gender.to(self.device)
                self.optimizer.zero_grad()  # zero all the gradient tensors
                predicted_emotion, predicted_speaker, predicted_gender = self.model(data)  # run forward step

                loss_emotion = self.criterion(predicted_emotion, target_emotion)  # compute loss
                loss_speaker = self.criterion(predicted_speaker, target_speaker)
                loss_gender = self.criterion(predicted_gender, target_gender)
                loss_total = loss_emotion + loss_speaker + loss_gender
                loss_total.backward()  # compute gradient tensors

                self.optimizer.step()  # update parameters

                epoch_loss_emotion += loss_emotion.item() * data.size(0)
                epoch_loss_speaker += loss_speaker.item() * data.size(0)
                epoch_loss_gender += loss_gender.item() * data.size(0)
                epoch_loss_total += loss_total.item() * data.size(0)  # compute the training loss value

                total += target[0].size(0)
                _, pred_labels_emotion = torch.max(predicted_emotion.data, 1)
                _, pred_labels_speaker = torch.max(predicted_speaker.data, 1)
                _, pred_labels_gender = torch.max(predicted_gender.data, 1)

                correct_emotion += (pred_labels_emotion == target_emotion).sum().item()
                correct_speaker += (pred_labels_speaker == target_speaker).sum().item()
                correct_gender += (pred_labels_gender == target_gender).sum().item()
            t = time.time() - t0

            epoch_loss_emotion /= dataset_size
            epoch_loss_gender /= dataset_size
            epoch_loss_speaker /= dataset_size

            train_acc_emotion = correct_emotion / total
            train_acc_speaker = correct_speaker / total
            train_acc_gender = correct_gender / total
            print('# Time passed: %.0f s' % t)
            print('# Epoch losses | emotion = {:.4f} | speaker = {:.4f} | gender = {:.4f} |'.format(
                epoch_loss_emotion, epoch_loss_speaker, epoch_loss_gender
            ))
            print('# Train accuracies | emotion = {} | speaker = {} | gender = {} |'.format(
                train_acc_emotion, train_acc_speaker, train_acc_gender
            ))


if __name__ == '__main__':
    pass
