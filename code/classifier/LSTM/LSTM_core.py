import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.utils.data as data
import os


class classification_LSTM_model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, LSTM_layer):
        super(classification_LSTM_model, self).__init__()
        torch.manual_seed(21368)
        torch.cuda.manual_seed(21368)
        self.LSTM_layer = nn.LSTM(input_size, hidden_size, LSTM_layer)
        self.FCNN = nn.Sequential(nn.Linear(hidden_size, 80),
                                  nn.Dropout(0.25),
                                  nn.ReLU(),
                                  nn.Linear(80, 10),
                                  nn.Dropout(0.3),
                                  nn.ReLU(),
                                  nn.Linear(10, output_size))

    def forward(self, input_x):
        LSTM_output, _ = self.LSTM_layer(input_x)
        seq_len, batch_size, h = LSTM_output.size()
        final_output = self.FCNN(LSTM_output.view(seq_len * batch_size, h))
        return final_output.view(seq_len, batch_size, -1)


class LSTM_utils():

    def __init__(self):
        self.init_learning_rate = 0.001
        self.max_epoch = 600
        self.max_learning_rate_drop_epoch = 10

    def train_LSTM(self, model, training_X, training_Y, batch_size):
        """
        :param model: LSTM model
        :param training_X: X in numpy array
        :param training_Y: Y in numpy array
        :return: CPU model, losses of each epoch, epoch index
        """
        criteria = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.init_learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        losses = []
        validate_epoch = []
        # transform to tensor
        train_X = torch.from_numpy(training_X)
        train_Y = torch.from_numpy(training_Y)
        # if GPU is available, move to GPU
        if torch.cuda.is_available():
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
            memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            device = torch.device('cuda:%s' % str(np.argmax(memory_gpu)))
        else:
            device = torch.device('cpu')
        model = model.to(device)
        train_X = train_X.float().to(device)
        train_Y = train_Y.float().to(device)
        # Batch
        dataset = data.TensorDataset(train_X, train_Y)
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        # Train LSTM
        model.train()
        bad_cnt = 0
        for epoch in range(self.max_epoch):
            batch_error = []
            for _, (batch_x, batch_y) in enumerate(dataloader):
                batch_x = batch_x.view(-1, 1, batch_x.size()[1])
                batch_y = batch_y.view(-1, 2)
                # FP
                train_output = model(batch_x)
                train_output = train_output.view(-1, 2)
                train_output = train_output.float()
                ty = torch.zeros(batch_y.size()[0], 1).to(device)
                ty[batch_y[:, 1] > 0.5, 0] = 1
                ty = ty.squeeze().long()
                loss = criteria(train_output, ty)
                batch_error.append(loss.item())
                # BP
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # schedule learning rate decrease
            loss = torch.mean(batch_size * torch.tensor(batch_error)).item()
            if bad_cnt == self.max_learning_rate_drop_epoch:
                scheduler.step()
                bad_cnt = 0
            if not len(losses) or loss < min(losses):
                print('Epoch %d get a better loss %.10f, model saved' % (epoch, loss))
                best_model = model
                bad_cnt = 0
            else:
                bad_cnt += 1
            losses.append(loss)
            validate_epoch.append(epoch)
        best_model = best_model.to('cpu')
        return best_model, losses, validate_epoch

    def test_LSTM(self, model, testing_X):
        """
        :param model: LSTM model
        :param testing_X: X in numpy array
        :return: the prob of class, numpy array in column
        """
        model.eval()
        # transform to tensor
        with torch.no_grad():
            test_X = torch.from_numpy(testing_X)
            test_X = test_X.view(-1, 1, test_X.size()[1])
            # if GPU is available, move to GPU
            if torch.cuda.is_available():
                os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
                memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
                device = torch.device('cuda:%s' % str(np.argmax(memory_gpu)))
            else:
                device = torch.device('cpu')
            model = model.to(device)
            test_X = test_X.float().to(device)
            # FP
            pred_result = torch.zeros(test_X.size()[0], 2).to('cpu')
            for test_sample_idx in range(test_X.size()[0]):
                this_sample = test_X[test_sample_idx, :, :]
                this_sample.unsqueeze_(1)
                pred_result[test_sample_idx, :] = nn.functional.softmax(model(this_sample), dim=2).view(-1, 2)
            pred = np.array(pred_result)
            model.train()
            return pred[:, 1]
