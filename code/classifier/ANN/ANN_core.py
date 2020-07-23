import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.utils.data as data
import os


class classification_ANN_model(nn.Module):

    def __init__(self, input_size, output_size):
        super(classification_ANN_model, self).__init__()
        torch.manual_seed(21368)
        torch.cuda.manual_seed(21368)
        self.FCNN = nn.Sequential(nn.Linear(input_size, 10),
                                  nn.Sigmoid(),
                                  nn.Linear(10, 10),
                                  nn.Sigmoid(),
                                  nn.Linear(10, 10),
                                  nn.Sigmoid(),
                                  nn.Linear(10, output_size),
                                  nn.Sigmoid())

    def forward(self, input_x):
        return self.FCNN(input_x)


class ANN_utils():

    def __init__(self):
        self.init_learning_rate = 0.001
        self.max_epoch = 1000
        self.max_learning_rate_drop_epoch = 10

    def train_ANN(self, model, training_X, training_Y, batch_size):
        """
        :param model: ANN model
        :param training_X: X in numpy array
        :param training_Y: Y in numpy array
        :return: CPU model, losses of each epoch, epoch index
        """
        criteria = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.init_learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)
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
        # Train ANN
        model.train()
        bad_cnt = 0
        for epoch in range(self.max_epoch):
            for _, (batch_x, batch_y) in enumerate(dataloader):
                # FP
                train_output = model(batch_x)
                loss = criteria(train_output, batch_y)
                # BP
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # validate
            output = torch.zeros(1, train_X.size()[0]).to(device)
            for train_sample_idx in range(train_X.size()[0]):
                output[0, train_sample_idx] = model(torch.unsqueeze(train_X[train_sample_idx, :], 0))
            loss = criteria(output, train_Y).item()
            # schedule learning rate decrease
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
        return best_model.to('cpu'), losses, validate_epoch

    def test_ANN(self, model, testing_X):
        """
        :param model: ANN model
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
            pred_result = torch.zeros(1, test_X.size()[0]).to(device)
            for test_sample_idx in range(test_X.size()[0]):
                pred_result[0, test_sample_idx] = model(torch.unsqueeze(test_X[test_sample_idx, :], 0))
            pred_result = pred_result.to('cpu')
            pred = np.array(pred_result)
            model.train()
            return pred
